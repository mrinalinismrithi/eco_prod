# EcoLens - AI Climate Intelligence Platform

## Project Overview
EcoLens is an AI-powered climate analysis platform that processes climate datasets, identifies environmental trends, and generates intelligent climate insights through a FastAPI backend.

The project combines data engineering, AI analysis, and backend development to help users understand climate change patterns across countries and regions.

---

## Main Objectives
- Analyze global climate data
- Identify warming trends
- Detect climate volatility
- Generate AI-powered summaries
- Provide structured climate analysis APIs

---

## Technologies Used
- Python
- Pandas
- FastAPI
- OpenAI API

---

## Project Architecture

Raw Climate Dataset
↓
ETL Pipeline
↓
Processed Climate Data
↓
AI Climate Analysis Agent
↓
FastAPI Backend

---

## Features

### 1. ETL Pipeline
- Loads raw climate datasets
- Cleans and processes data
- Handles missing values
- Generates processed datasets

### 2. Climate Analysis
- Detects fastest warming regions
- Calculates country volatility scores
- Identifies climate patterns
- Performs trend analysis

### 3. AI-Powered Insights
- Generates intelligent climate summaries
- Explains regional warming trends
- Produces structured analysis outputs

### 4. FastAPI Backend
- Provides API endpoints
- Returns JSON climate insights
- Handles analysis requests

---

## Run the App

From the project folder:

```bash
python run.py
```

Then open http://127.0.0.1:8001. FastAPI serves both the backend API and the frontend from the same port.


---

## Current Outputs

### Example Analysis Output
```json
{
  "region": "Africa",
  "fastest_warming_region": "Africa",
  "top_countries": [
    {
      "country": "Egypt",
      "volatility_score": 0.70
    }
  ]
}

