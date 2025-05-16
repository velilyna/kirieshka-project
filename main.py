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

def stock_price_forecast(x, y, degree, forecast_periods=12):
    """
    Uses least squares method to forecast future stock prices
    """
    coeffs = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coeffs)
    
    # Forecast future periods
    last_x = max(x)
    forecast_x = np.array([last_x + i + 1 for i in range(forecast_periods)])
    forecast_y = polynomial(forecast_x)
    
    # Full range for plotting
    x_full = np.append(x, forecast_x)
    y_full = np.append(y, forecast_y)
    
    return coeffs, polynomial, x_full, y_full, forecast_x, forecast_y

def portfolio_optimization(returns, risk_free_rate=0.02):
    """
    Calculate optimal portfolio weights using Lagrange multipliers
    """
    n = len(returns.columns)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov_matrix.values)
    
    # Calculate the vector of 1's
    ones = np.ones(n)
    
    # Calculate A, B, C, D from modern portfolio theory
    A = np.dot(np.dot(mean_returns.values, inv_cov), ones)
    B = np.dot(np.dot(ones, inv_cov), ones)
    C = np.dot(np.dot(mean_returns.values, inv_cov), mean_returns.values)
    D = B * C - A * A
    
    # Calculate the weights for the minimum variance portfolio
    min_var_weights = np.dot(inv_cov, ones) / B
    
    # Calculate the weights for the tangency portfolio
    tangency_weights = np.dot(inv_cov, mean_returns.values - risk_free_rate * ones) / np.dot(inv_cov, mean_returns.values - risk_free_rate * ones).sum()
    
    # Calculate the expected return and risk for different portfolios
    expected_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
    risks = []
    
    for er in expected_returns:
        lam = (C - er * A) / D
        mu = (er * B - A) / D
        weights = np.dot(inv_cov, (lam * ones + mu * mean_returns.values))
        
        portfolio_risk = np.sqrt(np.dot(np.dot(weights, cov_matrix.values), weights))
        risks.append(portfolio_risk)
    
    return min_var_weights, tangency_weights, expected_returns, risks

def yield_curve_interpolation(maturities, yields, new_maturities):
    """
    Uses cubic spline interpolation to create a smooth yield curve
    """
    # Sort by maturity
    idx = np.argsort(maturities)
    maturities = maturities[idx]
    yields = yields[idx]
    
    # Create cubic spline interpolation
    tck = interpolate.splrep(maturities, yields, s=0)
    
    # Interpolate at new maturities
    new_yields = interpolate.splev(new_maturities, tck, der=0)
    
    return new_maturities, new_yields, tck

def main():
    st.title("Financial Data Analysis Suite")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Loan Calculator", "Stock Price Forecast", "Portfolio Optimization", "Yield Curve Analysis"])
    
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
                
            # Calculate monthly payment for summary stats
            if unknown_param == "payment":
                monthly_payment = result
            else:
                monthly_payment = calculate_loan_parameter([principal, interest_rate, term, 0], "payment")
                st.info(f"Monthly Payment: ${monthly_payment:.2f}")
            
            # Only show total payment info if we have valid values
            if monthly_payment > 0 and term > 0:
                total_payment = monthly_payment * term
                total_interest = total_payment - principal
                st.info(f"Total Payment: ${total_payment:.2f} | Total Interest: ${total_interest:.2f}")
    
    with tab2:
        st.header("Stock Price Forecast (Least Squares Method)")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with historical prices", type="csv", key="stock_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                date_col = st.selectbox("Date Column", df.columns)
                price_col = st.selectbox("Price Column", df.columns)
                
                # Convert dates to numeric values for fitting
                df['date_numeric'] = pd.to_numeric(pd.to_datetime(df[date_col]))
                df['date_numeric'] = (df['date_numeric'] - df['date_numeric'].min()) / 86400000000000  # Convert to days
                
                x = df['date_numeric'].values
                y = df[price_col].values
                
                # Store dates for display
                dates = pd.to_datetime(df[date_col])
        else:
            st.subheader("Sample Stock Data")
            sample_data = st.text_area("Enter historical stock data (format: YYYY-MM-DD,price on each line)", 
                                       "2023-01-01,150.25\n2023-02-01,155.50\n2023-03-01,153.75\n2023-04-01,160.00\n"
                                       "2023-05-01,158.25\n2023-06-01,165.50\n2023-07-01,170.75\n2023-08-01,175.00\n"
                                       "2023-09-01,172.50\n2023-10-01,180.25\n2023-11-01,185.00\n2023-12-01,190.50")
            
            if sample_data:
                try:
                    data_lines = sample_data.strip().split("\n")
                    data_points = [line.split(",") for line in data_lines]
                    dates = pd.to_datetime([point[0] for point in data_points])
                    prices = np.array([float(point[1]) for point in data_points])
                    
                    # Convert dates to numeric for fitting
                    dates_numeric = pd.to_numeric(dates)
                    dates_numeric = (dates_numeric - dates_numeric.min()) / 86400000000000  # Convert to days
                    
                    x = dates_numeric
                    y = prices
                    
                    data_df = pd.DataFrame({"Date": dates, "Price": prices})
                    st.dataframe(data_df)
                except:
                    st.error("Invalid data format. Please use 'YYYY-MM-DD,price' on each line.")
                    x, y = None, None
        
        if 'x' in locals() and 'y' in locals() and x is not None and y is not None:
            degree = st.slider("Polynomial Degree", 1, 5, 2, 
                               help="Higher degrees can fit data better but may overfit")
            forecast_periods = st.slider("Forecast periods (months)", 1, 24, 6)
            
            coeffs, polynomial, x_full, y_full, forecast_x, forecast_y = stock_price_forecast(x, y, degree, forecast_periods)
            
            # Convert forecast x values back to dates for display
            if 'dates' in locals():
                last_date = dates.iloc[-1]
                forecast_dates = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_periods)]
                
                formula = f"Price = {polynomial}"
                st.subheader("Forecast Model")
                st.text(formula)
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    "Date": forecast_dates,
                    "Forecasted Price": forecast_y
                })
                
                st.subheader("Price Forecast")
                st.dataframe(forecast_df)
                
                # Plot the results
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot historical data
                ax.scatter(dates, y, color='blue', label='Historical Prices')
                
                # Plot forecast
                ax.scatter(forecast_dates, forecast_y, color='red', label='Forecasted Prices')
                
                # Plot trendline
                all_dates = dates.tolist() + forecast_dates
                ax.plot(all_dates, y_full, 'g--', label=f'Trend (degree {degree})')
                
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calculate metrics
                mse = np.mean((polynomial(x) - y) ** 2)
                last_price = y[-1]
                next_price = forecast_y[0]
                growth_rate = (next_price - last_price) / last_price * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Next Period Price", f"${next_price:.2f}", f"{growth_rate:.2f}%")
                with col2:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
    
    with tab3:
        st.header("Portfolio Optimization (Lagrange Multipliers)")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], key="portfolio_input", horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with asset returns", type="csv", key="portfolio_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                df = df.set_index(df.columns[0]) if len(df.columns) > 1 else df
                st.dataframe(df.head())
                returns = df
        else:
            st.subheader("Sample Portfolio Data")
            st.write("Monthly returns for 5 assets over 3 years (2020-2022)")
            
            # Create sample data
            np.random.seed(42)
            dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
            assets = ['Stock A', 'Stock B', 'Bond A', 'Bond B', 'REIT']
            
            # Generate returns with different characteristics
            returns_data = {
                'Stock A': np.random.normal(0.01, 0.05, len(dates)),
                'Stock B': np.random.normal(0.008, 0.04, len(dates)),
                'Bond A': np.random.normal(0.003, 0.01, len(dates)),
                'Bond B': np.random.normal(0.004, 0.015, len(dates)),
                'REIT': np.random.normal(0.006, 0.03, len(dates))
            }
            
            returns = pd.DataFrame(returns_data, index=dates)
            st.dataframe(returns.head())
        
        if 'returns' in locals() and returns is not None:
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0) / 100
            
            # Calculate portfolio statistics
            min_var_weights, tangency_weights, expected_returns, risks = portfolio_optimization(returns, risk_free_rate)
            
            # Display results
            st.subheader("Portfolio Analysis")
            
            # Create dataframe with asset statistics
            asset_stats = pd.DataFrame({
                'Mean Return': returns.mean(),
                'Standard Deviation': returns.std(),
                'Min. Variance Weight': min_var_weights,
                'Tangency Portfolio Weight': tangency_weights
            })
            st.dataframe(asset_stats)
            
            # Plot efficient frontier
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot efficient frontier
            ax.plot(risks, expected_returns, 'b-', label='Efficient Frontier')
            
            # Plot individual assets
            for i, asset in enumerate(returns.columns):
                ax.scatter(returns[asset].std(), returns[asset].mean(), 
                           label=asset, s=100)
            
            # Plot minimum variance portfolio
            min_var_return = np.dot(min_var_weights, returns.mean())
            min_var_risk = np.sqrt(np.dot(np.dot(min_var_weights, returns.cov()), min_var_weights))
            ax.scatter(min_var_risk, min_var_return, marker='*', color='g', s=200, label='Min. Variance')
            
            # Plot tangency portfolio
            tangency_return = np.dot(tangency_weights, returns.mean())
            tangency_risk = np.sqrt(np.dot(np.dot(tangency_weights, returns.cov()), tangency_weights))
            ax.scatter(tangency_risk, tangency_return, marker='*', color='r', s=200, label='Tangency')
            
            # Plot capital market line
            ax.plot([0, tangency_risk*1.5], [risk_free_rate, risk_free_rate + 1.5*(tangency_return-risk_free_rate)], 
                    'r--', label='Capital Market Line')
            
            ax.set_xlabel("Risk (Standard Deviation)")
            ax.set_ylabel("Expected Return")
            ax.set_title("Efficient Frontier and Optimal Portfolios")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Portfolio allocation
            st.subheader("Portfolio Allocation")
            
            portfolio_type = st.radio("Portfolio Type", ["Minimum Variance", "Tangency (Maximum Sharpe)"], horizontal=True)
            weights = min_var_weights if portfolio_type == "Minimum Variance" else tangency_weights
            
            # Display allocation pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(weights, labels=returns.columns, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
            
            # Display metrics
            portfolio_return = np.dot(weights, returns.mean())
            portfolio_risk = np.sqrt(np.dot(np.dot(weights, returns.cov()), weights))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Return", f"{portfolio_return*100:.2f}%")
            with col2:
                st.metric("Risk (Std Dev)", f"{portfolio_risk*100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    
    with tab4:
        st.header("Yield Curve Analysis (Cubic Spline)")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], key="yield_input", horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with yield curve data", type="csv", key="yield_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                maturity_col = st.selectbox("Maturity Column (years)", df.columns, key="yield_maturity")
                yield_col = st.selectbox("Yield Column (%)", df.columns, key="yield_rate")
                
                maturities = df[maturity_col].values
                yields = df[yield_col].values
        else:
            st.subheader("Sample Yield Curve Data")
            st.write("US Treasury yields by maturity (sample data)")
            
            sample_data = st.text_area("Enter data (format: maturity_years,yield_percent on each line)", 
                                     "0.25,3.95\n0.5,3.91\n1,3.85\n2,3.43\n3,3.25\n5,3.18\n7,3.20\n10,3.35\n20,3.70\n30,3.81",
                                     key="yield_sample")
            
            if sample_data:
                try:
                    data_lines = sample_data.strip().split("\n")
                    data_points = [line.split(",") for line in data_lines]
                    maturities = np.array([float(point[0]) for point in data_points])
                    yields = np.array([float(point[1]) for point in data_points])
                    
                    data_df = pd.DataFrame({"Maturity (Years)": maturities, "Yield (%)": yields})
                    st.dataframe(data_df)
                except:
                    st.error("Invalid data format. Please use 'maturity_years,yield_percent' on each line.")
                    maturities, yields = None, None
        
        if 'maturities' in locals() and 'yields' in locals() and maturities is not None and yields is not None:
            # Generate dense set of maturities for interpolation
            interp_points = st.slider("Number of interpolation points", 50, 500, 200)
            
            # Create interpolation
            dense_maturities = np.linspace(min(maturities), max(maturities), interp_points)
            dense_maturities, dense_yields, tck = yield_curve_interpolation(maturities, yields, dense_maturities)
            
            # Plot the yield curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(maturities, yields, color='blue', s=100, label='Observed Yields')
            ax.plot(dense_maturities, dense_yields, 'r-', label='Interpolated Yield Curve')
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Yield (%)")
            ax.set_title("Yield Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Calculate yield curve metrics
            st.subheader("Yield Curve Analysis")
            
            # Calculate slope (10y - 3m)
            short_idx = np.abs(maturities - 0.25).argmin()
            long_idx = np.abs(maturities - 10).argmin()
            slope = yields[long_idx] - yields[short_idx]
            
            # Calculate curvature (2*(2y) - (10y + 3m))
            mid_idx = np.abs(maturities - 2).argmin()
            curvature = 2 * yields[mid_idx] - (yields[long_idx] + yields[short_idx])
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Yield Curve Slope", f"{slope:.2f}%", 
                         "Normal" if slope > 0 else "Inverted (Recession Signal)")
            with col2:
                st.metric("Yield Curve Curvature", f"{curvature:.2f}%")
            
            # Calculate forward rates
            st.subheader("Forward Rate Analysis")
            
            # Spot rates to forward rates
            forward_rates = []
            forward_maturities = []
            
            for i in range(len(dense_maturities)-1):
                t1 = dense_maturities[i]
                t2 = dense_maturities[i+1]
                r1 = dense_yields[i]
                r2 = dense_yields[i+1]
                
                # Calculate forward rate
                forward_rate = ((1 + r2/100) ** t2) / ((1 + r1/100) ** t1) - 1
                forward_rate = forward_rate * 100
                
                forward_rates.append(forward_rate)
                forward_maturities.append(t1)
            
            # Plot forward rates
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(forward_maturities, forward_rates, 'g-', label='Implied Forward Rates')
            ax.plot(dense_maturities, dense_yields, 'r--', label='Spot Yield Curve')
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Rate (%)")
            ax.set_title("Spot Yield Curve vs. Forward Rates")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Custom yield calculation
            st.subheader("Custom Maturity Interpolation")
            col1, col2 = st.columns(2)
            with col1:
                custom_maturity = st.number_input("Enter maturity (years)", 
                                                 value=1.5,
                                                 min_value=float(min(maturities)),
                                                 max_value=float(max(maturities)))
            
            interpolated_yield = float(interpolate.splev(custom_maturity, tck, der=0))
            
            with col2:
                st.metric(f"Interpolated yield at {custom_maturity} years", f"{interpolated_yield:.4f}%")

    st.sidebar.title("Financial Analysis Suite")
    st.sidebar.write("This application demonstrates numerical methods in financial analysis:")
    st.sidebar.write("1. Newton's Method for loan calculations")
    st.sidebar.write("2. Least Squares Method for stock price forecasting")
    st.sidebar.write("3. Lagrange Multipliers for portfolio optimization")
    st.sidebar.write("4. Cubic Splines for yield curve analysis")

if __name__ == "__main__":
    main()
