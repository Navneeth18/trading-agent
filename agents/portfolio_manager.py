"""Portfolio Manager using DeepSeek-R1 with reasoning capabilities."""
import requests
import re
from typing import Dict
import config
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PortfolioManager:
    """Makes final trade decisions using DeepSeek-R1's reasoning capabilities."""
    
    def __init__(self):
        """Initialize Ollama client for DeepSeek-R1."""
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.PORTFOLIO_MANAGER_MODEL
    
    def extract_thinking(self, response: str) -> tuple[str, str]:
        """Extract <think> block and final decision from response."""
        # Extract thinking block
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        
        # Remove thinking block to get final answer
        final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        return thinking, final_answer
    
    def make_decision(self, ticker: str, sentiment_data: Dict, technical_data: Dict, 
                     market_data: Dict) -> Dict:
        """Make final trading decision with reasoning."""
        
        prompt = f"""You are a Portfolio Manager making critical trading decisions. Use your reasoning capabilities to validate this trade signal.

TICKER: {ticker}

SENTIMENT ANALYSIS:
- Overall Sentiment: {sentiment_data['avg_sentiment']}
- Sentiment Score: {sentiment_data['avg_score']:.2f} (-1 to 1 scale)
- Positive: {sentiment_data['positive_ratio']:.1%}
- Negative: {sentiment_data['negative_ratio']:.1%}
- Neutral: {sentiment_data['neutral_ratio']:.1%}
- Headlines Analyzed: {sentiment_data['total_headlines']}

TECHNICAL ANALYSIS:
- RSI: {technical_data['indicators']['rsi']:.2f}
- MACD: {technical_data['indicators']['macd']['macd']:.4f}
- MACD Signal: {technical_data['indicators']['macd']['signal']:.4f}
- MACD Histogram: {technical_data['indicators']['macd']['histogram']:.4f}
- Specialist Analysis: {technical_data['analysis']}

MARKET DATA:
- Current Price: ${market_data['current_price']:.2f}
- Change: ${market_data['change']:.2f} ({market_data['percent_change']:.2f}%)

INSTRUCTIONS:
1. Use <think> tags to show your Chain of Thought reasoning
2. In your <think> block, analyze:
   - Alignment between sentiment and technical signals
   - Risk factors and potential conflicts
   - Market conditions and price action
   - Why you accept or reject this signal
3. After </think>, provide your final decision in this exact format:

DECISION: [BUY/SELL/HOLD]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [One sentence explanation]

Be decisive and clear. Your reasoning in <think> must justify your final decision."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180,  # Increased to 3 minutes for DeepSeek-R1
                verify=False  # Bypass SSL verification
            )
            response.raise_for_status()
            full_response = response.json()['response']
            
            # Extract thinking and decision
            thinking, final_answer = self.extract_thinking(full_response)
            
            # Parse decision
            decision_match = re.search(r'DECISION:\s*(BUY|SELL|HOLD)', final_answer, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', final_answer, re.IGNORECASE)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', final_answer, re.IGNORECASE)
            
            decision = decision_match.group(1).upper() if decision_match else 'HOLD'
            confidence = confidence_match.group(1).upper() if confidence_match else 'LOW'
            reasoning = reasoning_match.group(1).strip() if reasoning_match else 'No reasoning provided'
            
            approved = decision in ['BUY', 'SELL']
            
            return {
                'ticker': ticker,
                'decision': decision,
                'confidence': confidence,
                'reasoning': reasoning,
                'thinking_process': thinking,
                'approved': approved,
                'full_response': full_response
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'decision': 'HOLD',
                'confidence': 'LOW',
                'reasoning': f'Error in decision making: {str(e)}',
                'thinking_process': '',
                'approved': False,
                'full_response': ''
            }
