"""Simplified LangGraph workflow: Data Ingestion -> Sentiment -> Technical -> Portfolio Manager."""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
from agents.sentiment_analyst import SentimentAnalyst
from agents.technical_specialist import TechnicalSpecialist
from agents.portfolio_manager import PortfolioManager
from data.finnhub_client import FinnhubClient
from data.yfinance_client import YFinanceClient
from database.db_manager import DatabaseManager
import config


class TradingState(TypedDict):
    """State for the trading workflow."""
    ticker: str
    market_data: Dict
    price_history: Dict
    headlines: List[str]
    sentiment_data: Dict
    technical_data: Dict
    decision: Dict
    error: str


class TradingWorkflow:
    """Simplified trading workflow using LangGraph."""
    
    def __init__(self):
        """Initialize workflow components."""
        self.sentiment_analyst = SentimentAnalyst()
        self.technical_specialist = TechnicalSpecialist()
        self.portfolio_manager = PortfolioManager()
        self.finnhub = FinnhubClient()
        self.yfinance = YFinanceClient()
        self.db = DatabaseManager()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(TradingState)
        
        # Add nodes
        workflow.add_node("data_ingestion", self.data_ingestion_node)
        workflow.add_node("sentiment_analysis", self.sentiment_analysis_node)
        workflow.add_node("technical_analysis", self.technical_analysis_node)
        workflow.add_node("portfolio_manager", self.portfolio_manager_node)
        
        # Define edges: Data -> Sentiment -> Technical -> Portfolio Manager -> END
        workflow.set_entry_point("data_ingestion")
        workflow.add_edge("data_ingestion", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "technical_analysis")
        workflow.add_edge("technical_analysis", "portfolio_manager")
        workflow.add_edge("portfolio_manager", END)
        
        return workflow.compile()
    
    def data_ingestion_node(self, state: TradingState) -> TradingState:
        """Node 1: Fetch market data and news."""
        ticker = state['ticker']
        
        try:
            # Fetch real-time quote from Finnhub
            market_data = self.finnhub.get_quote(ticker)
            if not market_data:
                state['error'] = f"Failed to fetch market data for {ticker}"
                return state
            
            # Fetch news headlines from yfinance
            headlines = self.yfinance.get_news(ticker, limit=10)
            
            # Fetch price history for technical analysis
            price_history = self.yfinance.get_price_history(ticker, period="1mo")
            
            state['market_data'] = market_data
            state['headlines'] = headlines
            state['price_history'] = price_history
            
        except Exception as e:
            state['error'] = f"Data ingestion error: {str(e)}"
        
        return state
    
    def sentiment_analysis_node(self, state: TradingState) -> TradingState:
        """Node 2: Analyze sentiment using FinBERT."""
        if state.get('error'):
            return state
        
        try:
            ticker = state['ticker']
            headlines = state['headlines']
            
            sentiment_data = self.sentiment_analyst.analyze_news(ticker, headlines)
            state['sentiment_data'] = sentiment_data
            
        except Exception as e:
            state['error'] = f"Sentiment analysis error: {str(e)}"
        
        return state
    
    def technical_analysis_node(self, state: TradingState) -> TradingState:
        """Node 3: Compute technical indicators using Llama 3.2."""
        if state.get('error'):
            return state
        
        try:
            ticker = state['ticker']
            market_data = state['market_data']
            price_history = state['price_history']
            
            if price_history.empty:
                state['error'] = "No price history available for technical analysis"
                return state
            
            technical_data = self.technical_specialist.analyze(ticker, market_data, price_history)
            state['technical_data'] = technical_data
            
        except Exception as e:
            state['error'] = f"Technical analysis error: {str(e)}"
        
        return state
    
    def portfolio_manager_node(self, state: TradingState) -> TradingState:
        """Node 4: Make final decision using DeepSeek-R1 with historical context."""
        if state.get('error'):
            return state
        
        try:
            ticker = state['ticker']
            sentiment_data = state['sentiment_data']
            technical_data = state['technical_data']
            market_data = state['market_data']
            
            # Fetch historical trades for this ticker
            historical_trades = self.db.get_recent_trades(ticker=ticker, limit=5)
            
            decision = self.portfolio_manager.make_decision(
                ticker, sentiment_data, technical_data, market_data, historical_trades
            )
            
            # Store trade in database
            self.db.insert_trade(
                ticker=ticker,
                action=decision['decision'],
                price=market_data['current_price'],
                quantity=0,  # Quantity calculation can be added
                reasoning=decision['reasoning'],
                sentiment_avg=sentiment_data['avg_score'],
                rsi=technical_data['indicators']['rsi'],
                macd=technical_data['indicators']['macd']['macd'],
                approved=decision['approved'],
                portfolio_manager_reasoning=decision['thinking_process']
            )
            
            state['decision'] = decision
            
        except Exception as e:
            state['error'] = f"Portfolio manager error: {str(e)}"
        
        return state
    
    def run(self, ticker: str) -> Dict:
        """Execute the workflow for a single ticker."""
        initial_state = TradingState(
            ticker=ticker,
            market_data={},
            price_history=None,
            headlines=[],
            sentiment_data={},
            technical_data={},
            decision={},
            error=""
        )
        
        final_state = self.graph.invoke(initial_state)
        return final_state
    
    def run_batch(self, tickers: List[str] = None) -> Dict[str, Dict]:
        """Execute workflow for multiple tickers."""
        if tickers is None:
            tickers = config.STOCKS
        
        results = {}
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            result = self.run(ticker)
            results[ticker] = result
        
        return results
