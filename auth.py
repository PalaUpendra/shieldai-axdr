from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, jwt_required,
    get_jwt_identity, get_jwt
)
from database import db, User, OTPStore, AuditLog
from otp_service import send_otp
from datetime import datetime, timedelta
import bcrypt

auth_bp = Blueprint('auth', __name__)

def log_action(username, action, details=None):
    db.session.add(AuditLog(
        username=username,
        action=action,
        ip_address=request.remote_addr,
        details=details
    ))
    db.session.commit()

# ── Step 1: Verify password ───────────────────────────
@auth_bp.route('/api/login', methods=['POST'])
def login():
    data     = request.json or {}
    username = data.get('username','').strip()
    password = data.get('password','').strip()
    method   = data.get('otp_method', None)

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = User.query.filter_by(username=username, is_active=True).first()
    if not user or not user.check_password(password):
        log_action(username, "LOGIN_FAILED", "Invalid credentials")
        return jsonify({"error": "Invalid credentials"}), 401

    otp_method = method or user.otp_method

    if otp_method == 'email' and not user.email:
        return jsonify({"error": "No email set for this account"}), 400
    if otp_method == 'sms' and not user.phone:
        return jsonify({"error": "No phone set for this account"}), 400

    otp, actual_method = send_otp(user, otp_method)
    if not otp:
        return jsonify({"error": f"Failed to send OTP via {otp_method}. Check email/phone is set."}), 500

    OTPStore.query.filter_by(username=username, used=False).delete()
    db.session.add(OTPStore(
        username   = username,
        otp_code   = otp,
        method     = actual_method,
        expires_at = datetime.utcnow() + timedelta(minutes=5)
    ))
    db.session.commit()

    log_action(username, "OTP_SENT", f"Via {actual_method}")

    hint = None
    if 'email' in actual_method and user.email and '@' in user.email:
        hint = user.email[:3] + "***" + user.email[user.email.index('@'):]

    fallback = (actual_method != otp_method)

    return jsonify({
        "message":    f"OTP sent via {actual_method}",
        "otp_method": actual_method,
        "username":   username,
        "hint":       hint,
        "fallback":   fallback
    })

# ── Step 2: Verify OTP ────────────────────────────────
@auth_bp.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    data     = request.json or {}
    username = data.get('username','').strip()
    otp_code = data.get('otp','').strip()

    if not username or not otp_code:
        return jsonify({"error": "Username and OTP required"}), 400

    record = OTPStore.query.filter_by(
        username=username, otp_code=otp_code, used=False
    ).first()

    if not record:
        log_action(username, "OTP_FAILED", "Invalid OTP")
        return jsonify({"error": "Invalid OTP"}), 401

    if datetime.utcnow() > record.expires_at:
        log_action(username, "OTP_EXPIRED")
        return jsonify({"error": "OTP expired — please login again"}), 401

    record.used = True
    user = User.query.filter_by(username=username).first()
    user.last_login = datetime.utcnow()
    db.session.commit()

    token = create_access_token(
        identity=username,
        additional_claims={"role": user.role, "name": user.name}
    )
    log_action(username, "LOGIN_SUCCESS", f"Role: {user.role}")

    return jsonify({
        "token":    token,
        "username": username,
        "role":     user.role,
        "name":     user.name
    })

@auth_bp.route('/api/logout', methods=['POST'])
@jwt_required()
def logout():
    log_action(get_jwt_identity(), "LOGOUT")
    return jsonify({"message": "Logged out"})

@auth_bp.route('/api/me')
@jwt_required()
def me():
    claims = get_jwt()
    return jsonify({
        "username": get_jwt_identity(),
        "role":     claims.get('role'),
        "name":     claims.get('name')
    })

@auth_bp.route('/api/users')
@jwt_required()
def list_users():
    if get_jwt().get('role') != 'admin':
        return jsonify({"error": "Admin only"}), 403
    return jsonify([u.to_dict() for u in User.query.all()])

@auth_bp.route('/api/users', methods=['POST'])
@jwt_required()
def create_user():
    if get_jwt().get('role') != 'admin':
        return jsonify({"error": "Admin only"}), 403
    data = request.json or {}
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Username already exists"}), 400
    hashed = bcrypt.hashpw(data['password'].encode(), bcrypt.gensalt()).decode()
    user = User(
        username   = data['username'],
        password   = hashed,
        role       = data.get('role', 'viewer'),
        name       = data.get('name', data['username']),
        email      = data.get('email'),
        phone      = data.get('phone'),
        otp_method = data.get('otp_method', 'email')
    )
    db.session.add(user)
    db.session.commit()
    log_action(get_jwt_identity(), "USER_CREATED", f"Created {data['username']}")
    return jsonify(user.to_dict()), 201

@auth_bp.route('/api/audit')
@jwt_required()
def audit_logs():
    if get_jwt().get('role') not in ['admin', 'analyst']:
        return jsonify({"error": "Not authorized"}), 403
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(100).all()
    return jsonify([l.to_dict() for l in logs])
