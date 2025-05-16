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

def main():
    st.title("Financial Analysis Suite")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Loan Calculator", "Investment ROI", "Bond Valuation", "Retirement Planning"])
    
    with tab1:
        st.header("Loan Calculator")
        st.write("Calculate unknown loan parameters using Newton's method")
        
        col1, col2 = st.columns(2)
        with col1:
            principal = st.number_input("Loan Amount ($)", min_value=1000.0, value=100000.0, step=1000.0)
            interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.1, value=5.0, step=0.1)
        with col2:
            term = st.number_input("Loan Term (months)", min_value=1, value=60, step=1)
            payment = st.number_input("Monthly Payment ($)", min_value=0.0, value=0.0, step=100.0)
        
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
        st.header("Investment ROI Projection")
        st.write("Project investment returns using polynomial trend fitting")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with investment data", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                x_col = st.selectbox("Time Period Column", df.columns)
                y_col = st.selectbox("Value/Return Column", df.columns)
                
                x = np.arange(len(df[x_col]))
                y = df[y_col].values
        else:
            st.subheader("Sample Investment Data")
            sample_data = st.text_area("Enter data (format: period,value on each line)", 
                                        "0,10000\n1,10700\n2,11200\n3,11900\n4,12500\n5,13200")
            
            if sample_data:
                try:
                    data_lines = sample_data.strip().split("\n")
                    data_points = [line.split(",") for line in data_lines]
                    x = np.array([float(point[0]) for point in data_points])
                    y = np.array([float(point[1]) for point in data_points])
                    
                    data_df = pd.DataFrame({"Period": x, "Value": y})
                    st.dataframe(data_df)
                except:
                    st.error("Invalid data format. Please use 'period,value' on each line.")
                    x, y = None, None
        
        if 'x' in locals() and 'y' in locals() and x is not None and y is not None:
            degree = st.slider("Trend Complexity", 1, 5, 2, help="Higher values create more complex projections")
            projection_periods = st.slider("Projection Periods", 1, 10, 3)
            
            coeffs = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coeffs)
            
            # Predict future values
            future_x = np.array([max(x) + i + 1 for i in range(projection_periods)])
            future_y = polynomial(future_x)
            
            # Full range for plotting
            full_x = np.append(x, future_x)
            full_y = np.append(y, polynomial(future_x))
            
            # Calculate ROI
            initial_investment = y[0]
            final_value = future_y[-1]
            total_roi = (final_value - initial_investment) / initial_investment * 100
            annual_roi = ((final_value / initial_investment) ** (1 / (len(full_x) - 1)) - 1) * 100
            
            # Show projection table
            projection_df = pd.DataFrame({
                "Period": future_x,
                "Projected Value": future_y.round(2)
            })
            st.subheader("Investment Projection")
            st.dataframe(projection_df)
            
            # Plot the results
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, color='blue', label='Historical Values')
            ax.scatter(future_x, future_y, color='red', label='Projected Values')
            ax.plot(full_x, polynomial(full_x), 'g--', label='Trend Line')
            ax.set_xlabel("Period")
            ax.set_ylabel("Value ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Show ROI metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total ROI", f"{total_roi:.2f}%")
            with col2:
                st.metric("Annualized ROI", f"{annual_roi:.2f}%")
    
    with tab3:
        st.header("Bond Valuation & Yield Curve")
        st.write("Analyze bond prices and interpolate yields across different maturities")
        
        method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], key="bond_input", horizontal=True)
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with bond data", type="csv", key="bond_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                maturity_col = st.selectbox("Maturity Column (years)", df.columns)
                yield_col = st.selectbox("Yield Column (%)", df.columns)
                
                maturities = df[maturity_col].values
                yields = df[yield_col].values
        else:
            st.subheader("Sample Bond Data")
            sample_data = st.text_area("Enter data (format: maturity_years,yield_percent on each line)", 
                                       "0.5,3.91\n1,3.85\n2,3.43\n3,3.25\n5,3.18\n7,3.20\n10,3.35\n30,3.81",
                                       key="bond_sample")
            
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
            # Create interpolation
            tck = interpolate.splrep(maturities, yields, s=0)
            
            # Create dense set of maturities for smooth curve
            dense_maturities = np.linspace(min(maturities), max(maturities), 100)
            dense_yields = interpolate.splev(dense_maturities, tck, der=0)
            
            # Bond price calculation
            st.subheader("Bond Price Calculator")
            col1, col2, col3 = st.columns(3)
            with col1:
                face_value = st.number_input("Face Value ($)", min_value=100.0, value=1000.0, step=100.0)
            with col2:
                coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, value=4.0, step=0.25)
            with col3:
                maturity = st.number_input("Time to Maturity (years)", min_value=0.5, value=5.0, step=0.5)
            
            # Find market yield for this maturity using interpolation
            market_yield = float(interpolate.splev(maturity, tck, der=0))
            
            # Calculate bond price
            coupon_payment = face_value * (coupon_rate / 100)
            periods = int(maturity * 2)  # Assuming semi-annual payments
            discount_rate = market_yield / 100 / 2  # Semi-annual rate
            
            present_value_coupons = coupon_payment / 2 * (1 - 1 / (1 + discount_rate) ** periods) / discount_rate
            present_value_principal = face_value / (1 + discount_rate) ** periods
            bond_price = present_value_coupons + present_value_principal
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Yield", f"{market_yield:.2f}%")
            with col2:
                st.metric("Bond Price", f"${bond_price:.2f}")
            
            # Plot yield curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(maturities, yields, color='blue', s=100, label='Market Yields')
            ax.plot(dense_maturities, dense_yields, 'r-', label='Yield Curve')
            ax.axvline(x=maturity, color='green', linestyle='--', label=f'Selected Maturity ({maturity} yrs)')
            ax.axhline(y=market_yield, color='green', linestyle='--')
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Yield (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Calculate yield curve metrics
            st.subheader("Yield Curve Analysis")
            
            # Calculate slope (10y - 2y or closest available)
            short_idx = np.abs(maturities - 2).argmin()
            long_idx = np.abs(maturities - 10).argmin()
            slope = yields[long_idx] - yields[short_idx]
            
            st.metric("Yield Curve Slope", f"{slope:.2f}%", 
                     "Normal" if slope > 0 else "Inverted (Recession Signal)")
    
    with tab4:
        st.header("Retirement Planning Calculator")
        st.write("Plan your retirement savings with custom contributions and returns")
        
        col1, col2 = st.columns(2)
        with col1:
            current_age = st.number_input("Current Age", min_value=18, max_value=70, value=30)
            retirement_age = st.number_input("Retirement Age", min_value=current_age+1, max_value=100, value=65)
            initial_savings = st.number_input("Current Savings ($)", min_value=0, value=10000)
        
        with col2:
            monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=500)
            annual_return = st.number_input("Expected Annual Return (%)", min_value=0.0, value=7.0, step=0.5)
            inflation_rate = st.number_input("Expected Inflation Rate (%)", min_value=0.0, value=2.5, step=0.5)
            
        # Calculate retirement statistics
        years_to_retirement = retirement_age - current_age
        months_to_retirement = years_to_retirement * 12
        real_return_rate = (1 + annual_return/100) / (1 + inflation_rate/100) - 1
        monthly_real_return = (1 + real_return_rate) ** (1/12) - 1
        
        # Calculate future value of current savings
        future_savings = initial_savings * (1 + real_return_rate) ** years_to_retirement
        
        # Calculate future value of monthly contributions
        if monthly_real_return > 0:
            future_contributions = monthly_contribution * ((1 + monthly_real_return) ** months_to_retirement - 1) / monthly_real_return
        else:
            future_contributions = monthly_contribution * months_to_retirement
            
        total_retirement_savings = future_savings + future_contributions
        
        # Calculate retirement income
        withdrawal_rate = 4.0  # 4% rule
        annual_retirement_income = total_retirement_savings * (withdrawal_rate / 100)
        monthly_retirement_income = annual_retirement_income / 12
        
        # Create data for visualization
        ages = np.arange(current_age, retirement_age + 1)
        savings_growth = np.zeros(len(ages))
        savings_growth[0] = initial_savings
        
        for i in range(1, len(ages)):
            savings_growth[i] = savings_growth[i-1] * (1 + real_return_rate) + monthly_contribution * 12
        
        # Display results
        st.subheader("Retirement Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total at Retirement", f"${total_retirement_savings:,.2f}")
        with col2:
            st.metric("Annual Income", f"${annual_retirement_income:,.2f}")
        with col3:
            st.metric("Monthly Income", f"${monthly_retirement_income:,.2f}")
        
        # Plot retirement growth
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ages, savings_growth, 'b-', marker='o')
        ax.set_xlabel("Age")
        ax.set_ylabel("Savings ($)")
        ax.set_title("Retirement Savings Growth")
        ax.grid(True, alpha=0.3)
        
        for i in range(0, len(ages), max(1, len(ages) // 6)):
            ax.annotate(f"${savings_growth[i]:,.0f}", 
                       (ages[i], savings_growth[i]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
            
        st.pyplot(fig)
        
        # Calculate contribution impact
        st.subheader("Contribution Analysis")
        
        # Calculate how much more would be saved with $100 more monthly
        additional_monthly = 100
        additional_savings = additional_monthly * ((1 + monthly_real_return) ** months_to_retirement - 1) / monthly_real_return if monthly_real_return > 0 else additional_monthly * months_to_retirement
        
        st.write(f"Adding ${additional_monthly} more per month would increase your retirement savings by: ${additional_savings:,.2f}")
        
        # Show what happens if starting 5 years earlier
        if current_age >= 23:
            extra_years = 5
            extra_months = extra_years * 12
            earlier_savings = monthly_contribution * ((1 + monthly_real_return) ** (months_to_retirement + extra_months) - 1) / monthly_real_return if monthly_real_return > 0 else monthly_contribution * (months_to_retirement + extra_months)
            difference = earlier_savings - future_contributions
            
            st.write(f"If you had started {extra_years} years earlier, you would have an additional: ${difference:,.2f}")

    st.sidebar.title("Financial Analysis Suite")
    st.sidebar.write("This application demonstrates financial calculations:")
    st.sidebar.write("1. Loan Calculator - Find loan parameters using Newton's method")
    st.sidebar.write("2. Investment ROI - Project returns using data fitting")
    st.sidebar.write("3. Bond Valuation - Price bonds and analyze yield curves")
    st.sidebar.write("4. Retirement Planning - Calculate retirement savings")

if __name__ == "__main__":
    main()
