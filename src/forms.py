from wtforms import EmailField, PasswordField, SubmitField, FileField
from wtforms.validators import DataRequired, Length, EqualTo, InputRequired
from flask_wtf import FlaskForm

class LoginForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired(), Length(8, 30)])
    submit = SubmitField("Login")

class SignUpForm(FlaskForm):
    email = EmailField("Email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired(), Length(8, 30)])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), Length(8, 30), EqualTo("password", message="Passwords do not match")])
    submit = SubmitField("Sign Up")


class UploadForm(FlaskForm):
    pdf = FileField('Upload PDF', validators=[DataRequired()])
    submit = SubmitField("Submit")