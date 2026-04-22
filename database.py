from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bcrypt

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    password      = db.Column(db.String(200), nullable=False)
    role          = db.Column(db.String(20),  default='viewer')
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=True)
    phone         = db.Column(db.String(20),  unique=True, nullable=True)
    otp_method    = db.Column(db.String(10),  default='email')
    is_active     = db.Column(db.Boolean,     default=True)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)
    last_login    = db.Column(db.DateTime,    nullable=True)

    def check_password(self, password):
        return bcrypt.checkpw(password.encode(), self.password.encode())

    def to_dict(self):
        return {
            "id":         self.id,
            "username":   self.username,
            "role":       self.role,
            "name":       self.name,
            "email":      self.email,
            "phone":      self.phone,
            "otp_method": self.otp_method,
            "is_active":  self.is_active,
            "created_at": str(self.created_at),
            "last_login": str(self.last_login) if self.last_login else None
        }

class OTPStore(db.Model):
    __tablename__ = 'otp_store'
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  nullable=False)
    otp_code   = db.Column(db.String(6),   nullable=False)
    method     = db.Column(db.String(10),  nullable=False)
    expires_at = db.Column(db.DateTime,    nullable=False)
    used       = db.Column(db.Boolean,     default=False)
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  nullable=False)
    action     = db.Column(db.String(200), nullable=False)
    ip_address = db.Column(db.String(50),  nullable=True)
    details    = db.Column(db.String(500), nullable=True)
    timestamp  = db.Column(db.DateTime,    default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":        self.id,
            "username":  self.username,
            "action":    self.action,
            "ip":        self.ip_address,
            "details":   self.details,
            "timestamp": str(self.timestamp)
        }

def init_db(app):
    with app.app_context():
        db.create_all()
        # Seed default users only if table is empty
        if User.query.count() == 0:
            defaults = [
                ("admin",   "admin123",   "admin",   "Admin User",  "palaupendra163@gmail.com", "+919999999999"),
                ("analyst", "analyst123", "analyst", "SOC Analyst", "analyst@shieldai.com",      "+918888888888"),
                ("viewer",  "viewer123",  "viewer",  "Viewer",      "viewer@shieldai.com",       "+917777777777"),
            ]
            for username, pwd, role, name, email, phone in defaults:
                # Skip if username already exists (safe re-run)
                if User.query.filter_by(username=username).first():
                    continue
                hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
                db.session.add(User(
                    username=username, password=hashed,
                    role=role, name=name,
                    email=email, phone=phone,
                    otp_method='email'
                ))
            db.session.commit()
            print("Default users created ✅")
            print("  admin   / admin123")
            print("  analyst / analyst123")
            print("  viewer  / viewer123")
        print(f"Database ready ✅  (SQLite — shieldai.db)")
