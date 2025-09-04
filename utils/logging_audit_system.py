#!/usr/bin/env python3
"""
Comprehensive Logging and Audit Trail System for Content Moderation
Provides structured logging, audit trails, security monitoring, and compliance reporting
"""

import os
import json
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import uuid
from contextlib import contextmanager
from functools import wraps

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logging.warning("SQLite not available")

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    logging.warning("Elasticsearch not available. Install with: pip install elasticsearch")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Install with: pip install psutil")

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AuditEventType(Enum):
    """Audit event type enumeration"""
    API_REQUEST = "api_request"
    MODEL_PREDICTION = "model_prediction"
    CACHE_ACCESS = "cache_access"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    ERROR_EVENT = "error_event"
    PERFORMANCE_EVENT = "performance_event"

class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

@dataclass
class AuditEvent:
    """Audit event entry"""
    event_id: str
    timestamp: str
    event_type: AuditEventType
    actor: str  # User, system, API client, etc.
    action: str  # What was done
    resource: str  # What was affected
    outcome: str  # Success, failure, etc.
    details: Dict[str, Any]
    security_level: SecurityLevel = SecurityLevel.LOW
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

class CustomFormatter(logging.Formatter):
    """Custom log formatter with structured output"""
    
    def __init__(self, include_extra: bool = True, json_format: bool = False):
        self.include_extra = include_extra
        self.json_format = json_format
        
        if json_format:
            super().__init__()
        else:
            format_str = (
                "%(asctime)s | %(levelname)-8s | %(name)-20s | "
                "%(funcName)-15s:%(lineno)-4d | %(message)s"
            )
            super().__init__(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        if self.json_format:
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                level=LogLevel(record.levelname),
                message=record.getMessage(),
                logger_name=record.name,
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                thread_id=str(threading.get_ident()),
                process_id=os.getpid(),
                session_id=getattr(record, 'session_id', None),
                user_id=getattr(record, 'user_id', None),
                request_id=getattr(record, 'request_id', None),
                extra_data={
                    k: v for k, v in record.__dict__.items()
                    if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                               'filename', 'module', 'lineno', 'funcName', 'created',
                               'msecs', 'relativeCreated', 'thread', 'threadName',
                               'processName', 'process', 'message', 'exc_info', 'exc_text',
                               'stack_info', 'session_id', 'user_id', 'request_id']
                } if self.include_extra else {}
            )
            return log_entry.to_json()
        else:
            formatted = super().format(record)
            if self.include_extra and hasattr(record, 'extra_data'):
                formatted += f" | Extra: {record.extra_data}"
            return formatted

class FileHandler(logging.FileHandler):
    """Enhanced file handler with rotation and compression"""
    
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False,
                 max_bytes=10*1024*1024, backup_count=5):
        super().__init__(filename, mode, encoding, delay)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._check_rotation()
    
    def emit(self, record):
        """Emit a record with rotation check"""
        if self.should_rotate():
            self.rotate()
        super().emit(record)
    
    def should_rotate(self) -> bool:
        """Check if file should be rotated"""
        if not os.path.exists(self.baseFilename):
            return False
        return os.path.getsize(self.baseFilename) >= self.max_bytes
    
    def rotate(self):
        """Rotate log files"""
        self.close()
        
        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            old_name = f"{self.baseFilename}.{i}"
            new_name = f"{self.baseFilename}.{i + 1}"
            if os.path.exists(old_name):
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
        
        # Move current file to .1
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, f"{self.baseFilename}.1")
        
        # Reopen with new file
        self.stream = self._open()
    
    def _check_rotation(self):
        """Check if rotation is needed on startup"""
        if self.should_rotate():
            self.rotate()

class DatabaseAuditHandler:
    """Database handler for audit events"""
    
    def __init__(self, db_path: str = "audit_trail.db"):
        if not SQLITE_AVAILABLE:
            raise ImportError("SQLite not available")
        
        self.db_path = db_path
        self._init_database()
        self._lock = threading.RLock()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    security_level TEXT NOT NULL,
                    session_id TEXT,
                    request_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actor ON audit_events(actor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_security_level ON audit_events(security_level)")
            
            conn.commit()
    
    def log_event(self, event: AuditEvent) -> bool:
        """Log audit event to database"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO audit_events (
                            event_id, timestamp, event_type, actor, action, resource,
                            outcome, security_level, session_id, request_id,
                            ip_address, user_agent, details
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id, event.timestamp, event.event_type.value,
                        event.actor, event.action, event.resource, event.outcome,
                        event.security_level.value, event.session_id, event.request_id,
                        event.ip_address, event.user_agent, json.dumps(event.details)
                    ))
                    conn.commit()
                return True
            except Exception as e:
                logging.error(f"Failed to log audit event: {e}")
                return False
    
    def query_events(self, start_time: Optional[str] = None, end_time: Optional[str] = None,
                    event_type: Optional[AuditEventType] = None, actor: Optional[str] = None,
                    security_level: Optional[SecurityLevel] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        with self._lock:
            try:
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)
                
                if actor:
                    query += " AND actor = ?"
                    params.append(actor)
                
                if security_level:
                    query += " AND security_level = ?"
                    params.append(security_level.value)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(query, params)
                    return [dict(row) for row in cursor.fetchall()]
            
            except Exception as e:
                logging.error(f"Failed to query audit events: {e}")
                return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Total events
                    cursor.execute("SELECT COUNT(*) FROM audit_events")
                    total_events = cursor.fetchone()[0]
                    
                    # Events by type
                    cursor.execute("""
                        SELECT event_type, COUNT(*) as count 
                        FROM audit_events 
                        GROUP BY event_type
                    """)
                    events_by_type = dict(cursor.fetchall())
                    
                    # Events by security level
                    cursor.execute("""
                        SELECT security_level, COUNT(*) as count 
                        FROM audit_events 
                        GROUP BY security_level
                    """)
                    events_by_security = dict(cursor.fetchall())
                    
                    # Recent activity (last 24 hours)
                    cursor.execute("""
                        SELECT COUNT(*) FROM audit_events 
                        WHERE timestamp >= datetime('now', '-1 day')
                    """)
                    recent_events = cursor.fetchone()[0]
                    
                    return {
                        'total_events': total_events,
                        'events_by_type': events_by_type,
                        'events_by_security_level': events_by_security,
                        'recent_events_24h': recent_events,
                        'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
                    }
            
            except Exception as e:
                logging.error(f"Failed to get audit statistics: {e}")
                return {}

class ElasticsearchHandler:
    """Elasticsearch handler for log aggregation and search"""
    
    def __init__(self, hosts: List[str] = ['localhost:9200'], 
                 index_prefix: str = 'contentmod-logs',
                 username: Optional[str] = None, password: Optional[str] = None):
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("Elasticsearch not available. Install with: pip install elasticsearch")
        
        self.index_prefix = index_prefix
        self.es_client = Elasticsearch(
            hosts=hosts,
            basic_auth=(username, password) if username and password else None
        )
        
        try:
            # Test connection
            self.es_client.info()
            logging.info("Connected to Elasticsearch")
        except Exception as e:
            logging.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def _get_index_name(self, date: Optional[str] = None) -> str:
        """Get index name with date suffix"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        return f"{self.index_prefix}-{date}"
    
    def index_log(self, log_entry: LogEntry) -> bool:
        """Index log entry to Elasticsearch"""
        try:
            index_name = self._get_index_name()
            doc = log_entry.to_dict()
            
            self.es_client.index(
                index=index_name,
                document=doc
            )
            return True
        except Exception as e:
            logging.error(f"Failed to index log to Elasticsearch: {e}")
            return False
    
    def search_logs(self, query: Dict[str, Any], 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   size: int = 100) -> List[Dict[str, Any]]:
        """Search logs in Elasticsearch"""
        try:
            # Build index pattern
            if start_date and end_date:
                # This is simplified - in production, you'd want proper date range handling
                index_pattern = f"{self.index_prefix}-*"
            else:
                index_pattern = self._get_index_name()
            
            response = self.es_client.search(
                index=index_pattern,
                query=query,
                size=size,
                sort=[{"timestamp": {"order": "desc"}}]
            )
            
            return [hit['_source'] for hit in response['hits']['hits']]
        
        except Exception as e:
            logging.error(f"Failed to search logs in Elasticsearch: {e}")
            return []

class SecurityMonitor:
    """Security monitoring and alerting"""
    
    def __init__(self, audit_handler: DatabaseAuditHandler):
        self.audit_handler = audit_handler
        self.alert_thresholds = {
            'failed_requests_per_minute': 10,
            'high_security_events_per_hour': 5,
            'unique_ips_per_minute': 50,
            'error_rate_threshold': 0.1  # 10%
        }
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for security alerts"""
        self.alert_callbacks.append(callback)
    
    def check_security_threats(self) -> List[Dict[str, Any]]:
        """Check for potential security threats"""
        threats = []
        current_time = datetime.now(timezone.utc)
        
        # Check for excessive failed requests
        one_minute_ago = (current_time - timedelta(minutes=1)).isoformat()
        failed_events = self.audit_handler.query_events(
            start_time=one_minute_ago,
            limit=1000
        )
        
        failed_count = sum(1 for event in failed_events if 'failed' in event.get('outcome', '').lower())
        if failed_count > self.alert_thresholds['failed_requests_per_minute']:
            threats.append({
                'type': 'excessive_failed_requests',
                'severity': SecurityLevel.HIGH,
                'count': failed_count,
                'threshold': self.alert_thresholds['failed_requests_per_minute'],
                'timeframe': '1 minute'
            })
        
        # Check for high security events
        one_hour_ago = (current_time - timedelta(hours=1)).isoformat()
        high_security_events = self.audit_handler.query_events(
            start_time=one_hour_ago,
            security_level=SecurityLevel.HIGH,
            limit=1000
        )
        
        if len(high_security_events) > self.alert_thresholds['high_security_events_per_hour']:
            threats.append({
                'type': 'high_security_events',
                'severity': SecurityLevel.CRITICAL,
                'count': len(high_security_events),
                'threshold': self.alert_thresholds['high_security_events_per_hour'],
                'timeframe': '1 hour'
            })
        
        # Trigger alerts
        for threat in threats:
            self._trigger_alert(threat)
        
        return threats
    
    def _trigger_alert(self, threat: Dict[str, Any]):
        """Trigger security alert"""
        alert_message = f"Security Alert: {threat['type']} - {threat['count']} events (threshold: {threat['threshold']})"
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_message, threat)
            except Exception as e:
                logging.error(f"Alert callback failed: {e}")

class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, log_dir: str = "./logs", 
                 enable_database: bool = True,
                 enable_elasticsearch: bool = False,
                 elasticsearch_hosts: Optional[List[str]] = None):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logging
        self.logger = logging.getLogger('content_moderation')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter(json_format=False))
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handlers
        self._setup_file_handlers()
        
        # Database audit handler
        self.db_handler = None
        if enable_database and SQLITE_AVAILABLE:
            try:
                self.db_handler = DatabaseAuditHandler(str(self.log_dir / "audit_trail.db"))
                logging.info("Database audit handler initialized")
            except Exception as e:
                logging.error(f"Failed to initialize database audit handler: {e}")
        
        # Elasticsearch handler
        self.es_handler = None
        if enable_elasticsearch and ELASTICSEARCH_AVAILABLE:
            try:
                hosts = elasticsearch_hosts or ['localhost:9200']
                self.es_handler = ElasticsearchHandler(hosts=hosts)
                logging.info("Elasticsearch handler initialized")
            except Exception as e:
                logging.error(f"Failed to initialize Elasticsearch handler: {e}")
        
        # Security monitor
        self.security_monitor = None
        if self.db_handler:
            self.security_monitor = SecurityMonitor(self.db_handler)
            # Add default alert callback
            self.security_monitor.add_alert_callback(self._default_security_alert)
        
        # Context storage for request tracking
        self._context = threading.local()
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log levels"""
        # General log file
        general_handler = FileHandler(
            str(self.log_dir / "application.log"),
            max_bytes=50*1024*1024,  # 50MB
            backup_count=10
        )
        general_handler.setFormatter(CustomFormatter(json_format=True))
        general_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(general_handler)
        
        # Error log file
        error_handler = FileHandler(
            str(self.log_dir / "errors.log"),
            max_bytes=10*1024*1024,  # 10MB
            backup_count=5
        )
        error_handler.setFormatter(CustomFormatter(json_format=True))
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        # Security log file
        security_handler = FileHandler(
            str(self.log_dir / "security.log"),
            max_bytes=20*1024*1024,  # 20MB
            backup_count=10
        )
        security_handler.setFormatter(CustomFormatter(json_format=True))
        security_handler.setLevel(logging.WARNING)
        self.logger.addHandler(security_handler)
    
    def _default_security_alert(self, message: str, threat: Dict[str, Any]):
        """Default security alert handler"""
        self.logger.critical(f"SECURITY ALERT: {message}", extra={'threat_details': threat})
    
    def set_context(self, session_id: Optional[str] = None, 
                   user_id: Optional[str] = None, 
                   request_id: Optional[str] = None):
        """Set logging context for current thread"""
        self._context.session_id = session_id
        self._context.user_id = user_id
        self._context.request_id = request_id or str(uuid.uuid4())
    
    def get_context(self) -> Dict[str, Optional[str]]:
        """Get current logging context"""
        return {
            'session_id': getattr(self._context, 'session_id', None),
            'user_id': getattr(self._context, 'user_id', None),
            'request_id': getattr(self._context, 'request_id', None)
        }
    
    def log(self, level: LogLevel, message: str, **extra_data):
        """Log message with context"""
        context = self.get_context()
        
        # Add context to extra data
        log_extra = {
            'session_id': context['session_id'],
            'user_id': context['user_id'],
            'request_id': context['request_id'],
            'extra_data': extra_data
        }
        
        # Log to structured logger
        getattr(self.logger, level.value.lower())(message, extra=log_extra)
        
        # Index to Elasticsearch if available
        if self.es_handler:
            log_entry = LogEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                level=level,
                message=message,
                logger_name=self.logger.name,
                module='audit_logger',
                function='log',
                line_number=0,
                thread_id=str(threading.get_ident()),
                process_id=os.getpid(),
                session_id=context['session_id'],
                user_id=context['user_id'],
                request_id=context['request_id'],
                extra_data=extra_data
            )
            self.es_handler.index_log(log_entry)
    
    def audit(self, event_type: AuditEventType, actor: str, action: str, 
             resource: str, outcome: str, details: Dict[str, Any],
             security_level: SecurityLevel = SecurityLevel.LOW,
             ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Log audit event"""
        context = self.get_context()
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details,
            security_level=security_level,
            session_id=context['session_id'],
            request_id=context['request_id'],
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Log to database
        if self.db_handler:
            self.db_handler.log_event(event)
        
        # Log to structured logger
        self.log(
            LogLevel.INFO if security_level == SecurityLevel.LOW else LogLevel.WARNING,
            f"Audit: {actor} {action} {resource} - {outcome}",
            audit_event=event.to_dict()
        )
    
    @contextmanager
    def request_context(self, session_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       request_id: Optional[str] = None):
        """Context manager for request-scoped logging"""
        old_context = self.get_context()
        
        try:
            self.set_context(session_id, user_id, request_id)
            yield self
        finally:
            self.set_context(
                old_context['session_id'],
                old_context['user_id'],
                old_context['request_id']
            )
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        stats = {'database': {}, 'security': {}, 'system': {}}
        
        # Database statistics
        if self.db_handler:
            stats['database'] = self.db_handler.get_statistics()
        
        # Security statistics
        if self.security_monitor:
            threats = self.security_monitor.check_security_threats()
            stats['security'] = {
                'active_threats': len(threats),
                'threat_details': threats
            }
        
        # System statistics
        if PSUTIL_AVAILABLE:
            stats['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        
        return stats

def audit_decorator(event_type: AuditEventType, action: str, resource: str = "unknown"):
    """Decorator for automatic audit logging"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get audit logger from global or create one
            audit_logger = getattr(wrapper, '_audit_logger', None)
            if not audit_logger:
                audit_logger = AuditLogger()
            
            start_time = time.time()
            actor = kwargs.get('user_id', 'system')
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful execution
                audit_logger.audit(
                    event_type=event_type,
                    actor=actor,
                    action=action,
                    resource=resource,
                    outcome="success",
                    details={
                        'function': func.__name__,
                        'execution_time_ms': (time.time() - start_time) * 1000,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                return result
            
            except Exception as e:
                # Log failed execution
                audit_logger.audit(
                    event_type=AuditEventType.ERROR_EVENT,
                    actor=actor,
                    action=action,
                    resource=resource,
                    outcome="failure",
                    details={
                        'function': func.__name__,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'execution_time_ms': (time.time() - start_time) * 1000
                    },
                    security_level=SecurityLevel.MEDIUM
                )
                raise
        
        return wrapper
    return decorator

def create_logging_audit_demo():
    """
    Demonstrate comprehensive logging and audit capabilities
    """
    print("📋 Comprehensive Logging and Audit System Demo")
    print("=" * 60)
    
    try:
        # Initialize audit logger
        audit_logger = AuditLogger(
            log_dir="./demo_logs",
            enable_database=True,
            enable_elasticsearch=False  # Set to True if Elasticsearch is available
        )
        
        print("\n1. Basic Logging Demo")
        print("-" * 30)
        
        # Set context for request tracking
        with audit_logger.request_context(
            session_id="demo_session_123",
            user_id="demo_user",
            request_id="req_001"
        ):
            # Log different levels
            audit_logger.log(LogLevel.INFO, "Application started", component="main")
            audit_logger.log(LogLevel.DEBUG, "Processing request", endpoint="/api/moderate")
            audit_logger.log(LogLevel.WARNING, "High CPU usage detected", cpu_percent=85.2)
            audit_logger.log(LogLevel.ERROR, "Database connection failed", 
                           error="Connection timeout", retry_count=3)
        
        print("✅ Basic logging completed")
        
        print("\n2. Audit Trail Demo")
        print("-" * 30)
        
        # Log various audit events
        audit_events = [
            {
                'event_type': AuditEventType.API_REQUEST,
                'actor': 'user_123',
                'action': 'moderate_text',
                'resource': '/api/v1/moderate',
                'outcome': 'success',
                'details': {'text_length': 150, 'prediction': 'safe', 'confidence': 0.95},
                'security_level': SecurityLevel.LOW
            },
            {
                'event_type': AuditEventType.MODEL_PREDICTION,
                'actor': 'ml_model_v2',
                'action': 'predict',
                'resource': 'toxicity_classifier',
                'outcome': 'success',
                'details': {'model_version': '2.1', 'inference_time_ms': 45},
                'security_level': SecurityLevel.LOW
            },
            {
                'event_type': AuditEventType.SECURITY_EVENT,
                'actor': 'unknown_ip',
                'action': 'failed_authentication',
                'resource': '/api/auth',
                'outcome': 'failure',
                'details': {'ip_address': '192.168.1.100', 'attempts': 5},
                'security_level': SecurityLevel.HIGH,
                'ip_address': '192.168.1.100'
            },
            {
                'event_type': AuditEventType.CONFIGURATION_CHANGE,
                'actor': 'admin_user',
                'action': 'update_model_threshold',
                'resource': 'model_config',
                'outcome': 'success',
                'details': {'old_threshold': 0.7, 'new_threshold': 0.8},
                'security_level': SecurityLevel.MEDIUM
            }
        ]
        
        for event_data in audit_events:
            audit_logger.audit(**event_data)
            print(f"✅ Logged: {event_data['action']} by {event_data['actor']}")
        
        print("\n3. Decorator-based Audit Demo")
        print("-" * 30)
        
        @audit_decorator(AuditEventType.MODEL_PREDICTION, "classify_text")
        def classify_text(text: str, user_id: str = "system") -> Dict[str, Any]:
            """Example function with automatic audit logging"""
            time.sleep(0.1)  # Simulate processing
            return {
                'prediction': 'safe' if len(text) < 100 else 'review',
                'confidence': 0.92,
                'text_length': len(text)
            }
        
        # Set audit logger for decorator
        classify_text._audit_logger = audit_logger
        
        # Test the decorated function
        result1 = classify_text("This is a safe message", user_id="demo_user")
        result2 = classify_text("This is a longer message that might need review" * 3, user_id="demo_user")
        
        print(f"✅ Classification 1: {result1['prediction']}")
        print(f"✅ Classification 2: {result2['prediction']}")
        
        print("\n4. Security Monitoring Demo")
        print("-" * 30)
        
        if audit_logger.security_monitor:
            # Simulate some security events
            for i in range(3):
                audit_logger.audit(
                    event_type=AuditEventType.SECURITY_EVENT,
                    actor=f"suspicious_ip_{i}",
                    action="failed_request",
                    resource="/api/moderate",
                    outcome="failure",
                    details={'error': 'invalid_token'},
                    security_level=SecurityLevel.HIGH
                )
            
            # Check for threats
            threats = audit_logger.security_monitor.check_security_threats()
            print(f"✅ Security threats detected: {len(threats)}")
            
            for threat in threats:
                print(f"  - {threat['type']}: {threat['count']} events")
        
        print("\n5. Statistics and Reporting Demo")
        print("-" * 30)
        
        # Get comprehensive statistics
        stats = audit_logger.get_audit_statistics()
        
        print("📊 Audit Statistics:")
        if 'database' in stats and stats['database']:
            db_stats = stats['database']
            print(f"  Total Events: {db_stats.get('total_events', 0)}")
            print(f"  Recent Events (24h): {db_stats.get('recent_events_24h', 0)}")
            print(f"  Database Size: {db_stats.get('database_size_mb', 0):.2f} MB")
            
            if 'events_by_type' in db_stats:
                print("  Events by Type:")
                for event_type, count in db_stats['events_by_type'].items():
                    print(f"    {event_type}: {count}")
        
        if 'security' in stats and stats['security']:
            sec_stats = stats['security']
            print(f"  Active Security Threats: {sec_stats.get('active_threats', 0)}")
        
        if 'system' in stats and stats['system']:
            sys_stats = stats['system']
            print(f"  System CPU: {sys_stats.get('cpu_percent', 0):.1f}%")
            print(f"  System Memory: {sys_stats.get('memory_percent', 0):.1f}%")
        
        print("\n6. Query and Search Demo")
        print("-" * 30)
        
        if audit_logger.db_handler:
            # Query recent events
            recent_events = audit_logger.db_handler.query_events(limit=5)
            print(f"📋 Recent Events ({len(recent_events)}):")
            
            for event in recent_events[:3]:  # Show first 3
                print(f"  - {event['timestamp']}: {event['actor']} {event['action']} {event['resource']}")
            
            # Query by event type
            api_events = audit_logger.db_handler.query_events(
                event_type=AuditEventType.API_REQUEST,
                limit=10
            )
            print(f"\n🔍 API Request Events: {len(api_events)}")
        
        print("\n✅ Comprehensive logging and audit demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logging.error(f"Logging audit demo error: {e}")
    
    finally:
        # Cleanup demo files
        import shutil
        demo_dir = "./demo_logs"
        if os.path.exists(demo_dir):
            try:
                shutil.rmtree(demo_dir)
                print(f"\n🧹 Cleaned up {demo_dir}")
            except Exception as e:
                print(f"Failed to cleanup {demo_dir}: {e}")

if __name__ == "__main__":
    # Add missing import for timedelta
    from datetime import timedelta
    create_logging_audit_demo()