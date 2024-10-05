from wtforms import EmailField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
from flask_wtf import FlaskForm

class LoginForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired(), Length(8, 30)])
    submit = SubmitField("Login")

class SignUpForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired(), Length(8, 30)])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), Length(8, 30), EqualTo("password")])
    submit = SubmitField("Sign Up")