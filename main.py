import matplotlib
matplotlib.use('Agg')
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import io


st.set_page_config(layout="wide", page_title="Financial Analysis Suite")

def loan_equation(x, principal, payment, term):
    return principal * x * (1 + x) ** term / ((1 + x) ** term - 1) - payment

def loan_equation_derivative(x, principal, term):
    numerator = principal * ((1 + x) ** term - 1) * ((1 + x) ** term + term * x) - principal * x * term * (1 + x) ** (term - 1)
    denominator = ((1 + x) ** term - 1) ** 2
    return numerator / denominator

def newton_method(func, derivative, x0, args, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        f_val = func(x, *args)
        if abs(f_val) < tol:
            return x
        f_prime = derivative(x, args[0], args[2])
        x = x - f_val / f_prime
    return x

def calculate_loan_parameter(known_parameters, unknown_parameter):
    principal, interest_rate, term, payment = known_parameters
    
    if unknown_parameter == "interest_rate":
        x0 = 0.1 / 12
        monthly_rate = newton_method(loan_equation, loan_equation_derivative, x0, (principal, payment, term))
        return monthly_rate * 12 * 100
    
    elif unknown_parameter == "payment":
        monthly_rate = interest_rate / (12 * 100)
        payment = principal * monthly_rate * (1 + monthly_rate) ** term / ((1 + monthly_rate) ** term - 1)
        return payment
    
    elif unknown_parameter == "principal":
        monthly_rate = interest_rate / (12 * 100)
        principal = payment * ((1 + monthly_rate) ** term - 1) / (monthly_rate * (1 + monthly_rate) ** term)
        return principal
    
    elif unknown_parameter == "term":
        monthly_rate = interest_rate / (12 * 100)
        term = np.log(payment / (payment - principal * monthly_rate)) / np.log(1 + monthly_rate)
        return round(term)

def least_squares_approximation(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return coeffs

def predict_values(x, coeffs, prediction_points=10):
    polynomial = np.poly1d(coeffs)
    x_range = max(x) - min(x)
    x_pred = np.linspace(min(x), max(x) + 0.3 * x_range, prediction_points)
    y_pred = polynomial(x_pred)
    return x_pred, y_pred

def lagrange_basis_polynomial(x, x_points, j):
    basis = 1.0
    for i in range(len(x_points)):
        if i != j:
            basis *= (x - x_points[i]) / (x_points[j] - x_points[i])
    return basis

def lagrange_interpolate(x_points, y_points, x):
    result = 0.0
    for j in range(len(x_points)):
        result += y_points[j] * lagrange_basis_polynomial(x, x_points, j)
    return result

def apply_cubic_spline(x, y, points=500):
    tck = interpolate.splrep(x, y, s=0)
    x_new = np.linspace(min(x), max(x), points)
    y_new = interpolate.splev(x_new, tck, der=0)
    return x_new, y_new

def main():
    st.title("Financial Data Analysis Suite")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Loan Calculator", "Least Squares", "Lagrange Interpolation", "Cubic Spline"])
    
    with tab1:
        st.header("Loan Calculator (Newton's Method)")
        
        col1, col2 = st.columns(2)
        with col1:
            principal = st.number_input("Loan Amount", min_value=1000.0, value=100000.0, step=1000.0)
            interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.1, value=5.0, step=0.1)
        with col2:
            term = st.number_input("Loan Term (months)", min_value=1, value=60, step=1)
            payment = st.number_input("Monthly Payment", min_value=0.0, value=0.0, step=100.0)
        
        unknown_param = st.selectbox("Calculate", ["payment", "principal", "interest_rate", "term"])
        
        if st.button("Calculate", key="loan_calc"):
            known_params = [principal, interest_rate, term, payment]
            if unknown_param == "payment":
                known_params[3] = 0
            elif unknown_param == "principal":
                known_params[0] = 0
            elif unknown_param == "interest_rate":
                known_params[1] = 0
            elif unknown_param == "term":
                known_params[2] = 0
            
            result = calculate_loan_parameter(known_params, unknown_param)
            
            if unknown_param == "payment":
                st.success(f"Monthly Payment: ${result:.2f}")
            elif unknown_param == "principal":
                st.success(f"Loan Amount: ${result:.2f}")
            elif unknown_param == "interest_rate":
                st.success(f"Annual Interest Rate: {result:.2f}%")
            elif unknown_param == "term":
                st.success(f"Loan Term: {result:.0f} months ({result/12:.1f} years)")
                
            if unknown_param != "payment":
                monthly_payment = calculate_loan_parameter([principal, interest_rate, term, 0], "payment")
                st.info(f"Monthly Payment: ${monthly_payment:.2f}")
                
            total_payment = monthly_payment * term
            total_interest = total_payment - principal
            st.info(f"Total Payment: ${total_payment:.2f} | Total Interest: ${total_interest:.2f}")
    
    with tab2:
        st.header("Least Squares Method")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV data", type="csv", key="lsm_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                x_col = st.selectbox("X Column", df.columns)
                y_col = st.selectbox("Y Column", df.columns)
                
                x = df[x_col].values
                y = df[y_col].values
        else:
            st.subheader("Sample Data")
            sample_data = st.text_area("Enter data (format: x,y on each line)", 
                                        "1,2\n2,3\n3,5\n4,7\n5,11\n6,13\n7,17\n8,19")
            
            if sample_data:
                try:
                    data_lines = sample_data.strip().split("\n")
                    data_points = [line.split(",") for line in data_lines]
                    x = np.array([float(point[0]) for point in data_points])
                    y = np.array([float(point[1]) for point in data_points])
                    
                    data_df = pd.DataFrame({"x": x, "y": y})
                    st.dataframe(data_df)
                except:
                    st.error("Invalid data format. Please use 'x,y' on each line.")
                    x, y = None, None
        
        if 'x' in locals() and 'y' in locals() and x is not None and y is not None:
            degree = st.slider("Polynomial Degree", 1, 10, 2)
            prediction_points = st.number_input("Number of prediction points", 10, 500, 100)
            
            coeffs = least_squares_approximation(x, y, degree)
            x_pred, y_pred = predict_values(x, coeffs, prediction_points)
            
            polynomial = np.poly1d(coeffs)
            formula = f"y = {polynomial}"
            st.subheader("Polynomial Formula")
            st.text(formula)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, color='blue', label='Original Data')
            ax.plot(x_pred, y_pred, color='red', label=f'Polynomial (degree {degree})')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("Prediction")
            x_value = st.number_input("Enter X value for prediction", value=float(max(x))+1)
            predicted_y = polynomial(x_value)
            st.success(f"Predicted Y for X={x_value}: {predicted_y:.4f}")
            
            mse = np.mean((polynomial(x) - y) ** 2)
            st.info(f"Mean Squared Error: {mse:.6f}")
    
    with tab3:
        st.header("Lagrange Polynomial Interpolation")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], key="lagrange_input", horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV data", type="csv", key="lagrange_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                x_col = st.selectbox("X Column", df.columns, key="lagrange_x")
                y_col = st.selectbox("Y Column", df.columns, key="lagrange_y")
                
                x = df[x_col].values
                y = df[y_col].values
        else:
            st.subheader("Sample Data")
            sample_data = st.text_area("Enter data (format: x,y on each line)", 
                                        "1,3\n2,6\n4,24\n5,39\n7,81",
                                        key="lagrange_sample")
            
            if sample_data:
                try:
                    data_lines = sample_data.strip().split("\n")
                    data_points = [line.split(",") for line in data_lines]
                    x = np.array([float(point[0]) for point in data_points])
                    y = np.array([float(point[1]) for point in data_points])
                    
                    data_df = pd.DataFrame({"x": x, "y": y})
                    st.dataframe(data_df)
                except:
                    st.error("Invalid data format. Please use 'x,y' on each line.")
                    x, y = None, None
        
        if 'x' in locals() and 'y' in locals() and x is not None and y is not None:
            missing_x = st.number_input("X value to interpolate", 
                                       value=float(np.mean([min(x), max(x)])),
                                       min_value=float(min(x)),
                                       max_value=float(max(x)))
            
            interpolated_value = lagrange_interpolate(x, y, missing_x)
            
            x_dense = np.linspace(min(x), max(x), 1000)
            y_dense = [lagrange_interpolate(x, y, xi) for xi in x_dense]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, color='blue', label='Original Data')
            ax.scatter([missing_x], [interpolated_value], color='red', s=100, label='Interpolated Point')
            ax.plot(x_dense, y_dense, 'g--', label='Lagrange Polynomial')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.success(f"Interpolated value at x={missing_x}: {interpolated_value:.4f}")
            
            st.subheader("Missing Data Recovery")
            st.write("Select a point to remove and recover using interpolation:")
            point_to_remove = st.selectbox("Point to remove", range(len(x)), format_func=lambda i: f"X={x[i]}, Y={y[i]}")
            
            if st.button("Recover Missing Point"):
                x_missing = x[point_to_remove]
                y_missing = y[point_to_remove]
                
                x_reduced = np.delete(x, point_to_remove)
                y_reduced = np.delete(y, point_to_remove)
                
                y_recovered = lagrange_interpolate(x_reduced, y_reduced, x_missing)
                error = abs(y_recovered - y_missing)
                relative_error = error / abs(y_missing) * 100 if y_missing != 0 else float('inf')
                
                st.write(f"Actual value at X={x_missing}: {y_missing}")
                st.write(f"Recovered value: {y_recovered:.4f}")
                st.write(f"Absolute error: {error:.4f}")
                st.write(f"Relative error: {relative_error:.2f}%")
    
    with tab4:
        st.header("Cubic Spline Smoothing")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], key="spline_input", horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV data", type="csv", key="spline_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                x_col = st.selectbox("X Column", df.columns, key="spline_x")
                y_col = st.selectbox("Y Column", df.columns, key="spline_y")
                
                x = df[x_col].values
                y = df[y_col].values
        else:
            st.subheader("Sample Data")
            sample_data = st.text_area("Enter data (format: x,y on each line)", 
                                       "1,2.1\n1.5,3.5\n2,4.0\n2.5,3.8\n3,5.2\n3.5,6.1\n4,5.8\n4.5,7.2\n5,8.0\n5.5,7.9\n6,9.1",
                                       key="spline_sample")
            
            if sample_data:
                try:
                    data_lines = sample_data.strip().split("\n")
                    data_points = [line.split(",") for line in data_lines]
                    x = np.array([float(point[0]) for point in data_points])
                    y = np.array([float(point[1]) for point in data_points])
                    
                    data_df = pd.DataFrame({"x": x, "y": y})
                    st.dataframe(data_df)
                except:
                    st.error("Invalid data format. Please use 'x,y' on each line.")
                    x, y = None, None
        
        if 'x' in locals() and 'y' in locals() and x is not None and y is not None:
            points = st.slider("Number of interpolation points", 100, 1000, 500)
            
            x_new, y_new = apply_cubic_spline(x, y, points)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, color='blue', label='Original Data')
            ax.plot(x_new, y_new, color='red', label='Cubic Spline')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("Interpolation")
            interp_x = st.number_input("Enter X value for interpolation", 
                                      value=float(np.mean([min(x), max(x)])),
                                      min_value=float(min(x)),
                                      max_value=float(max(x)))
            
            tck = interpolate.splrep(x, y, s=0)
            interp_y = float(interpolate.splev(interp_x, tck, der=0))
            
            st.success(f"Interpolated value at x={interp_x}: {interp_y:.4f}")

    st.sidebar.title("Financial Analysis Suite")
    st.sidebar.write("This application demonstrates numerical methods in financial analysis:")
    st.sidebar.write("1. Newton's Method for loan calculations")
    st.sidebar.write("2. Least Squares Method for trend prediction")
    st.sidebar.write("3. Lagrange Interpolation for data recovery")
    st.sidebar.write("4. Cubic Splines for data smoothing")

if __name__ == "__main__":
    main()
