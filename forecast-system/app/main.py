from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import pandas as pd
from app.services.data_fetcher import ForecastDataFetcher
from app.services.forecast_predictor import ForecastPredictor
from app.services.visualizer import ForecastVisualizer
from config import TWELVE_DATA_API_KEY, FOREX_PAIRS, INTERVALS

app = FastAPI()
data_fetcher = ForecastDataFetcher(TWELVE_DATA_API_KEY)
predictor = ForecastPredictor()

class ForecastRequest(BaseModel):
    pair: str
    interval: str
    window_size: Optional[int] = 50
    forecast_size: Optional[int] = 10

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    """API endpoint to get price forecast"""
    if request.pair not in FOREX_PAIRS:
        raise HTTPException(status_code=400, detail="Unsupported currency pair")
    if request.interval not in INTERVALS:
        raise HTTPException(status_code=400, detail="Unsupported interval")
    
    try:
        # Fetch recent data
        data = data_fetcher.fetch_recent_for_forecast(
            request.pair, 
            request.interval, 
            request.window_size + 50
        )
        
        # Get forecast
        forecast = predictor.predict_future_prices(
            request.pair,
            request.interval,
            data,
            request.window_size,
            request.forecast_size
        )
        
        if not forecast["success"]:
            raise HTTPException(status_code=500, detail=forecast["error"])
            
        img_base64 = ForecastVisualizer.plot_forecast(data, forecast)
        
        return {
            "forecast": forecast,
            "visualization": img_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_forecast_ui():
    """Simple UI to visualize forecasts"""
    return """
    <html>
        <head>
            <title>Forex Price Forecast</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Forex Price Forecast</h1>
            <div>
                <label for="pair">Currency Pair:</label>
                <select id="pair">
                    <option value="EUR/USD">EUR/USD</option>
                    <!-- Add pairs here -->
                </select>
                
                <label for="interval">Interval:</label>
                <select id="interval">
                    <option value="15min">15min</option>
                    <option value="30min">30min</option>
                    <option value="1h">1h</option>
                </select>
                
                <button onclick="getForecast()">Get Forecast</button>
            </div>
            
            <div id="plot"></div>
            
            <script>
                function getForecast() {
                    const pair = document.getElementById("pair").value;
                    const interval = document.getElementById("interval").value;
                    
                    fetch("/api/forecast", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ 
                            pair, 
                            interval,
                            window_size: 50,
                            forecast_size: 10
                        })
                    })
                    .then(response => {
                        if (!response.ok) {
                            return response.json().then(err => Promise.reject(err));
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log("API Response:", data);
                        
                        if (!data.forecast || !data.forecast.forecast) {
                            throw new Error('Invalid forecast data: ' + JSON.stringify(data));
                        }
                        
                        // Historical trace
                        const historical_trace = {
                            x: data.forecast.historical_timestamps.map(ts => new Date(ts)),
                            y: data.forecast.historical_prices,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Historical Prices',
                            line: {color: 'blue'}
                        };
                        
                        // Forecast trace
                        const forecast_trace = {
                            x: data.forecast.forecast_timestamps.map(ts => new Date(ts)),
                            y: data.forecast.forecast,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Forecast',
                            line: {color: 'red', dash: 'dash'},
                            marker: {color: 'red'}
                        };
                        
                        // Current price marker
                        const current_trace = {
                            x: [new Date(data.forecast.last_historical_timestamp)],
                            y: [data.forecast.last_historical_price],
                            type: 'scatter',
                            mode: 'markers',
                            name: 'Current Price',
                            marker: {color: 'black', size: 6}
                        };
                        
                        const layout = {
                            title: `${pair} ${interval} Forecast`,
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Price' }
                        };
                        
                        Plotly.newPlot('plot', [historical_trace, forecast_trace, current_trace], layout);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        const errorMsg = error.message || 'Failed to get forecast';
                        document.getElementById("plot").innerHTML = 
                            `<div style="color: red;">Error: ${errorMsg}</div>`;
                    });
                }
            </script>
        </body>
    </html>
    """