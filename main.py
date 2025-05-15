import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

st.set_page_config(layout="wide", page_title="Financial Analysis Suite")

st.title("📊 Financial Analysis Suite")
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a module:", [
    "Loan Calculator",
    "Least Squares Forecast",
    "Lagrange Interpolation",
    "Cubic Spline Smoothing"
])

# ---------- Loan Calculator ---------- #
def loan_equation(x, principal, payment, term):
    return principal * x * (1 + x) ** term / ((1 + x) ** term - 1) - payment

def loan_equation_derivative(x, principal, term):
    numerator = principal * ((1 + x) ** term - 1) * ((1 + x) ** term + term * x) - principal * x * term * (1 + x) ** (term - 1)
    denominator = ((1 + x) ** term - 1) ** 2
    return numerator / denominator

def newton_method(func, derivative, x0, args, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        f_val = func(x, *args)
        if abs(f_val) < tol:
            return x
        f_prime = derivative(x, args[0], args[2])
        x = x - f_val / f_prime
    return x

if section == "Loan Calculator":
    st.header("Loan Calculator (Newton's Method)")
    known = st.selectbox("Which parameter to calculate?", ["Interest Rate", "Monthly Payment"])

    principal = st.number_input("Loan Amount (Principal)", min_value=1000.0, value=100000.0)
    term = st.number_input("Term (months)", min_value=1, value=120)

    if known == "Interest Rate":
        payment = st.number_input("Monthly Payment", min_value=100.0, value=1200.0)
        if st.button("Calculate Interest Rate"):
            x0 = 0.05 / 12
            rate = newton_method(loan_equation, loan_equation_derivative, x0, (principal, payment, term))
            st.success(f"Estimated Annual Interest Rate: {rate * 12 * 100:.2f}%")
    else:
        rate = st.number_input("Annual Interest Rate (%)", min_value=0.1, value=5.0)
        monthly_rate = rate / 12 / 100
        payment = principal * monthly_rate * (1 + monthly_rate) ** term / ((1 + monthly_rate) ** term - 1)
        st.success(f"Monthly Payment: ${payment:.2f}")

# ---------- Least Squares Approximation ---------- #
elif section == "Least Squares Forecast":
    st.header("Least Squares Forecast")
    uploaded_file = st.file_uploader("Upload CSV file with 'Date' and 'Close' columns", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        df.dropna(inplace=True)
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        degree = st.slider("Polynomial Degree", 1, 5, 2)
        coeffs = np.polyfit(df['Days'], df['Close'], degree)
        poly = np.poly1d(coeffs)

        df['Forecast'] = poly(df['Days'])

        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Close'], label='Original')
        ax.plot(df['Date'], df['Forecast'], label='Least Squares', linestyle='--')
        ax.set_title("Least Squares Approximation")
        ax.legend()
        st.pyplot(fig)

# ---------- Lagrange Interpolation ---------- #
elif section == "Lagrange Interpolation":
    st.header("Lagrange Polynomial Interpolation")
    uploaded_file = st.file_uploader("Upload CSV file with missing 'Close' values", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        known = df.dropna(subset=['Close'])
        missing = df[df['Close'].isna()]

        def lagrange_interp(x, xp, yp):
            result = 0
            for i in range(len(xp)):
                term = yp[i]
                for j in range(len(xp)):
                    if i != j:
                        term *= (x - xp[j]) / (xp[i] - xp[j])
                result += term
            return result

        interpolated = []
        for i, row in missing.iterrows():
            x = row['Days']
            value = lagrange_interp(x, known['Days'].values, known['Close'].values)
            interpolated.append(value)
            df.loc[i, 'Close'] = value

        st.write("Interpolated Data:", df)

        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Close'], label='Interpolated', marker='o')
        ax.set_title("Lagrange Interpolation")
        st.pyplot(fig)

# ---------- Cubic Spline Smoothing ---------- #
elif section == "Cubic Spline Smoothing":
    st.header("Cubic Spline Smoothing")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        df.dropna(inplace=True)
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        cs = interpolate.CubicSpline(df['Days'], df['Close'])
        df['Spline'] = cs(df['Days'])

        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Close'], label='Original', marker='o')
        ax.plot(df['Date'], df['Spline'], label='Cubic Spline', linestyle='--')
        ax.set_title("Cubic Spline Smoothing")
        ax.legend()
        st.pyplot(fig)
