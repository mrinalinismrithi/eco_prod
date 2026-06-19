from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class CountryInsight(BaseModel):
    country: str
    volatility_score: float = Field(..., description="Volatility or warming score")
    summary: str = Field(..., description="Short analytical insight")


class ClimateAnalysisResponse(BaseModel):
    """Structured response for country/region analysis"""
    region: str
    top_countries: List[CountryInsight]
    overall_summary: str
    live_weather_insight: Optional[str] = None


class WeatherComparison(BaseModel):
    """For comparing live weather with historical data"""
    city: str
    current_temp: float
    historical_avg: float
    difference: float
    insight: str


class SimpleResponse(BaseModel):
    """Flexible general response"""
    answer: str
    key_insights: Optional[List[str]] = None
    sources: Optional[List[str]] = Field(default=None, description="CSV or Live Weather")


class FastestWarmingResponse(BaseModel):
    """For fastest warming region queries"""
    fastest_region: str
    warming_rate: float
    insight: str
    live_comparison: Optional[str] = None


# Main flexible response model for the agent
class EcoLensResponse(BaseModel):
    """Main response model used by the AI Assistant"""
    response: str = Field(..., description="Main natural language answer")
    structured_data: Optional[Dict] = Field(default=None, description="Optional structured data")
    data_source: str = Field(..., description="CSV or Live_Weather or Both") 