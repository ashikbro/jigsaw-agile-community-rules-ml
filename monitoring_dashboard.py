#!/usr/bin/env python3
"""
Monitoring Dashboard for Content Moderation System
Real-time metrics visualization and system health monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import json
from datetime import datetime, timedelta
import psutil
import threading
from collections import deque, defaultdict
import numpy as np
from typing import Dict, List, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5  # seconds
MAX_DATA_POINTS = 100

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self):
        self.api_metrics = deque(maxlen=MAX_DATA_POINTS)
        self.system_metrics = deque(maxlen=MAX_DATA_POINTS)
        self.model_metrics = defaultdict(lambda: deque(maxlen=MAX_DATA_POINTS))
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = deque(maxlen=MAX_DATA_POINTS)
        self.running = False
        self.thread = None
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_loop, daemon=True)
            self.thread.start()
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collect_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect API health metrics
                self._collect_api_metrics()
                # Collect system metrics
                self._collect_system_metrics()
                time.sleep(REFRESH_INTERVAL)
            except Exception as e:
                st.error(f"Metrics collection error: {e}")
                time.sleep(REFRESH_INTERVAL)
    
    def _collect_api_metrics(self):
        """Collect API performance metrics"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms
            
            timestamp = datetime.now()
            
            if response.status_code == 200:
                health_data = response.json()
                
                metric = {
                    'timestamp': timestamp,
                    'response_time': response_time,
                    'status': 'healthy',
                    'models_loaded': health_data.get('models_loaded', 0),
                    'uptime': health_data.get('uptime', 0)
                }
                
                self.api_metrics.append(metric)
                self.response_times.append(response_time)
                self.request_counts['health'] += 1
            else:
                self.error_counts['api_error'] += 1
                
        except Exception as e:
            self.error_counts['connection_error'] += 1
            metric = {
                'timestamp': datetime.now(),
                'response_time': None,
                'status': 'error',
                'error': str(e)
            }
            self.api_metrics.append(metric)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metric = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
            
            self.system_metrics.append(metric)
            
        except Exception as e:
            st.error(f"System metrics collection error: {e}")
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API statistics"""
        if not self.api_metrics:
            return {}
        
        recent_metrics = list(self.api_metrics)[-10:]  # Last 10 metrics
        healthy_count = sum(1 for m in recent_metrics if m.get('status') == 'healthy')
        
        response_times = [m['response_time'] for m in recent_metrics if m.get('response_time')]
        
        return {
            'health_rate': (healthy_count / len(recent_metrics)) * 100 if recent_metrics else 0,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'total_requests': sum(self.request_counts.values()),
            'total_errors': sum(self.error_counts.values()),
            'models_loaded': recent_metrics[-1].get('models_loaded', 0) if recent_metrics else 0
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.system_metrics:
            return {}
        
        latest = list(self.system_metrics)[-1]
        return {
            'cpu_percent': latest.get('cpu_percent', 0),
            'memory_percent': latest.get('memory_percent', 0),
            'memory_used_gb': latest.get('memory_used_gb', 0),
            'memory_total_gb': latest.get('memory_total_gb', 0),
            'disk_percent': latest.get('disk_percent', 0),
            'disk_used_gb': latest.get('disk_used_gb', 0),
            'disk_total_gb': latest.get('disk_total_gb', 0)
        }

def create_response_time_chart(metrics_collector: MetricsCollector):
    """Create response time chart"""
    if not metrics_collector.response_times:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    times = list(metrics_collector.response_times)
    timestamps = [datetime.now() - timedelta(seconds=i*REFRESH_INTERVAL) 
                 for i in range(len(times)-1, -1, -1)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=times,
        mode='lines+markers',
        name='Response Time',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="API Response Time",
        xaxis_title="Time",
        yaxis_title="Response Time (ms)",
        height=300,
        showlegend=False
    )
    
    return fig

def create_system_metrics_chart(metrics_collector: MetricsCollector):
    """Create system metrics chart"""
    if not metrics_collector.system_metrics:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    metrics = list(metrics_collector.system_metrics)
    timestamps = [m['timestamp'] for m in metrics]
    cpu_data = [m['cpu_percent'] for m in metrics]
    memory_data = [m['memory_percent'] for m in metrics]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=cpu_data, name='CPU', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=memory_data, name='Memory', line=dict(color='#2ca02c')),
        row=2, col=1
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(range=[0, 100])
    
    return fig

def create_model_performance_chart():
    """Create model performance comparison chart"""
    # Mock data for model performance
    models = ['Logistic Regression', 'SVM', 'Random Forest', 'Neural Network']
    accuracy = [0.85, 0.82, 0.88, 0.91]
    precision = [0.83, 0.80, 0.86, 0.89]
    recall = [0.87, 0.84, 0.90, 0.93]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy, marker_color='#1f77b4'))
    fig.add_trace(go.Bar(name='Precision', x=models, y=precision, marker_color='#ff7f0e'))
    fig.add_trace(go.Bar(name='Recall', x=models, y=recall, marker_color='#2ca02c'))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Content Moderation Monitoring",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Content Moderation System - Monitoring Dashboard")
    st.markdown("Real-time monitoring of API performance, system resources, and model metrics")
    
    # Initialize metrics collector
    if 'metrics_collector' not in st.session_state:
        st.session_state.metrics_collector = MetricsCollector()
        st.session_state.metrics_collector.start_collection()
    
    metrics_collector = st.session_state.metrics_collector
    
    # Sidebar controls
    st.sidebar.header("⚙️ Dashboard Controls")
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 30, REFRESH_INTERVAL)
        time.sleep(refresh_rate)
        st.rerun()
    
    if st.sidebar.button("🔄 Manual Refresh"):
        st.rerun()
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current stats
    api_stats = metrics_collector.get_api_stats()
    system_stats = metrics_collector.get_system_stats()
    
    # Key metrics cards
    with col1:
        st.metric(
            "API Health",
            f"{api_stats.get('health_rate', 0):.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Avg Response Time",
            f"{api_stats.get('avg_response_time', 0):.1f}ms",
            delta=None
        )
    
    with col3:
        st.metric(
            "CPU Usage",
            f"{system_stats.get('cpu_percent', 0):.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            "Memory Usage",
            f"{system_stats.get('memory_percent', 0):.1f}%",
            delta=None
        )
    
    # Charts section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_response_time_chart(metrics_collector),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_system_metrics_chart(metrics_collector),
            use_container_width=True
        )
    
    # Model performance section
    st.markdown("---")
    st.plotly_chart(
        create_model_performance_chart(),
        use_container_width=True
    )
    
    # Detailed metrics tables
    st.markdown("---")
    st.subheader("📈 Detailed Metrics")
    
    tab1, tab2, tab3 = st.tabs(["API Metrics", "System Metrics", "Error Logs"])
    
    with tab1:
        if api_stats:
            st.json(api_stats)
        else:
            st.info("No API metrics available yet")
    
    with tab2:
        if system_stats:
            st.json(system_stats)
        else:
            st.info("No system metrics available yet")
    
    with tab3:
        if metrics_collector.error_counts:
            error_df = pd.DataFrame([
                {'Error Type': k, 'Count': v} 
                for k, v in metrics_collector.error_counts.items()
            ])
            st.dataframe(error_df, use_container_width=True)
        else:
            st.success("No errors recorded")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip**: Keep this dashboard open to monitor your content moderation system in real-time!"
    )

if __name__ == "__main__":
    main()