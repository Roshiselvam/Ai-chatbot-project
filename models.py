from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from extensions import db
from flask_login import UserMixin

class User(db.Model,UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)  # NOT UNIQUE anymore
    email = db.Column(db.String(120), unique=True, nullable=False)  # UNIQUE
    password = db.Column(db.String(200), nullable=False)

    sessions = db.relationship("ChatSession", backref="user", lazy=True)
    messages = db.relationship("ChatHistory", backref="user", lazy=True)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title = db.Column(db.String(200), nullable=False, default="New Chat")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    messages = db.relationship("ChatHistory", backref="session", lazy=True)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_session.id"), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
