#!/usr/bin/env python3
"""
System Resource Monitoring Utility
Monitors CPU, memory, disk usage and provides alerts
"""

import psutil
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import threading
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for older Python versions
    from email.MIMEText import MIMEText as MimeText
    from email.MIMEMultipart import MIMEMultipart as MimeMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resource_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Optional[List[float]] = None

@dataclass
class AlertThresholds:
    """Alert threshold configuration"""
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 75.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    network_threshold_mbps: float = 100.0

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, email_config: Optional[Dict] = None):
        self.email_config = email_config
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldown = {}
        self.cooldown_period = 300  # 5 minutes
    
    def send_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Send alert notification"""
        # Check cooldown
        now = datetime.now()
        if alert_type in self.alert_cooldown:
            if now - self.alert_cooldown[alert_type] < timedelta(seconds=self.cooldown_period):
                return
        
        self.alert_cooldown[alert_type] = now
        
        alert = {
            'timestamp': now,
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        logger.log(log_level, f"ALERT [{severity.upper()}] {alert_type}: {message}")
        
        # Send email if configured
        if self.email_config:
            self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"System Alert: {alert['type']} [{alert['severity'].upper()}]"
            
            body = f"""
            System Alert Notification
            
            Type: {alert['type']}
            Severity: {alert['severity'].upper()}
            Time: {alert['timestamp']}
            Message: {alert['message']}
            
            Please check your system immediately.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            if self.email_config.get('use_tls'):
                server.starttls()
            if self.email_config.get('username'):
                server.login(self.email_config['username'], self.email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert['type']}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert['timestamp'] > cutoff]

class ResourceMonitor:
    """Main resource monitoring class"""
    
    def __init__(self, 
                 thresholds: Optional[AlertThresholds] = None,
                 alert_manager: Optional[AlertManager] = None,
                 monitoring_interval: int = 30):
        self.thresholds = thresholds or AlertThresholds()
        self.alert_manager = alert_manager or AlertManager()
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.running = False
        self.monitor_thread = None
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[ResourceMetrics], None]):
        """Add callback function to be called on each metric collection"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.running:
            logger.warning("Monitoring is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring loop")
        
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and send alerts
                self._check_thresholds(metrics)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix-like systems only)
        load_average = None
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            # Windows doesn't have load average
            pass
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_count=process_count,
            load_average=load_average
        )
    
    def _check_thresholds(self, metrics: ResourceMetrics):
        """Check metrics against thresholds and send alerts"""
        # CPU alerts
        if metrics.cpu_percent >= self.thresholds.cpu_critical:
            self.alert_manager.send_alert(
                "cpu_critical",
                f"CPU usage critical: {metrics.cpu_percent:.1f}%",
                "critical"
            )
        elif metrics.cpu_percent >= self.thresholds.cpu_warning:
            self.alert_manager.send_alert(
                "cpu_warning",
                f"CPU usage high: {metrics.cpu_percent:.1f}%",
                "warning"
            )
        
        # Memory alerts
        if metrics.memory_percent >= self.thresholds.memory_critical:
            self.alert_manager.send_alert(
                "memory_critical",
                f"Memory usage critical: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB/{metrics.memory_total_gb:.1f}GB)",
                "critical"
            )
        elif metrics.memory_percent >= self.thresholds.memory_warning:
            self.alert_manager.send_alert(
                "memory_warning",
                f"Memory usage high: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB/{metrics.memory_total_gb:.1f}GB)",
                "warning"
            )
        
        # Disk alerts
        if metrics.disk_percent >= self.thresholds.disk_critical:
            self.alert_manager.send_alert(
                "disk_critical",
                f"Disk usage critical: {metrics.disk_percent:.1f}% ({metrics.disk_used_gb:.1f}GB/{metrics.disk_total_gb:.1f}GB)",
                "critical"
            )
        elif metrics.disk_percent >= self.thresholds.disk_warning:
            self.alert_manager.send_alert(
                "disk_warning",
                f"Disk usage high: {metrics.disk_percent:.1f}% ({metrics.disk_used_gb:.1f}GB/{metrics.disk_total_gb:.1f}GB)",
                "warning"
            )
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, hours: int = 1) -> List[ResourceMetrics]:
        """Get metrics history for specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff]
    
    def get_average_metrics(self, hours: int = 1) -> Optional[Dict]:
        """Get average metrics for specified period"""
        history = self.get_metrics_history(hours)
        if not history:
            return None
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in history) / len(history),
            'avg_memory_percent': sum(m.memory_percent for m in history) / len(history),
            'avg_disk_percent': sum(m.disk_percent for m in history) / len(history),
            'avg_process_count': sum(m.process_count for m in history) / len(history),
            'period_hours': hours,
            'sample_count': len(history)
        }
    
    def export_metrics(self, filename: str, hours: int = 24):
        """Export metrics to JSON file"""
        history = self.get_metrics_history(hours)
        
        # Convert to serializable format
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'metrics': [
                {
                    **asdict(m),
                    'timestamp': m.timestamp.isoformat()
                }
                for m in history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(history)} metrics to {filename}")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="System Resource Monitor")
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--cpu-warning', type=float, default=70.0, help='CPU warning threshold')
    parser.add_argument('--cpu-critical', type=float, default=90.0, help='CPU critical threshold')
    parser.add_argument('--memory-warning', type=float, default=75.0, help='Memory warning threshold')
    parser.add_argument('--memory-critical', type=float, default=90.0, help='Memory critical threshold')
    parser.add_argument('--export-interval', type=int, default=3600, help='Export interval in seconds')
    
    args = parser.parse_args()
    
    # Configure thresholds
    thresholds = AlertThresholds(
        cpu_warning=args.cpu_warning,
        cpu_critical=args.cpu_critical,
        memory_warning=args.memory_warning,
        memory_critical=args.memory_critical
    )
    
    # Create monitor
    monitor = ResourceMonitor(
        thresholds=thresholds,
        monitoring_interval=args.interval
    )
    
    # Add periodic export callback
    last_export = [datetime.now()]
    
    def export_callback(metrics: ResourceMetrics):
        if (datetime.now() - last_export[0]).seconds >= args.export_interval:
            filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            monitor.export_metrics(filename)
            last_export[0] = datetime.now()
    
    monitor.add_callback(export_callback)
    
    # Start monitoring
    try:
        monitor.start_monitoring()
        logger.info("Resource monitor started. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping resource monitor...")
        monitor.stop_monitoring()
        logger.info("Resource monitor stopped")

if __name__ == "__main__":
    main()