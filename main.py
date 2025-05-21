import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import io

st.set_page_config(layout="wide", page_title="Financial Analysis Suite")

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
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
        if payment <= principal * monthly_rate:
            return float('inf')
        term = np.log(payment / (payment - principal * monthly_rate)) / np.log(1 + monthly_rate)
        return round(term)

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
        if abs(f_prime) < 1e-10:
            f_prime = 1e-10
        x = x - f_val / f_prime
        if x <= 0:
            x = 0.0001
    return x

def process_bond_data(maturities, yields, maturity, face_value, coupon_rate):
    tck = interpolate.splrep(maturities, yields, s=0)
    market_yield = float(interpolate.splev(maturity, tck, der=0))
    
    coupon_payment = face_value * (coupon_rate / 100) 
    periods = int(maturity * 2) 
    discount_rate = market_yield / 100 / 2 
    
    present_value_coupons = coupon_payment / 2 * (1 - 1 / (1 + discount_rate) ** periods) / discount_rate if discount_rate > 0 else coupon_payment / 2 * periods
    present_value_principal = face_value / (1 + discount_rate) ** periods
    bond_price = present_value_coupons + present_value_principal
    
    return market_yield, bond_price, tck

def main():
    st.title("Financial Analysis Suite")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Loan Calculator", "Investment ROI", "Bond Valuation", "Retirement Planning"])
    
    with tab1:
        render_loan_calculator()
    
    with tab2:
        render_investment_roi()
    
    with tab3:
        render_bond_valuation()
        
    with tab4:
        render_retirement_planning()

    st.sidebar.title("Financial Analysis Suite")
    st.sidebar.write("This application demonstrates financial calculations:")
    st.sidebar.write("1. Loan Calculator - Find loan parameters using Newton's method")
    st.sidebar.write("2. Investment ROI - Project returns using data fitting")
    st.sidebar.write("3. Bond Valuation - Price bonds and analyze yield curves")
    st.sidebar.write("4. Retirement Planning - Calculate retirement savings")
    st.sidebar.markdown("---")
    st.sidebar.info("This app is for educational purposes only. Financial decisions should be made in consultation with qualified professionals.")

def render_loan_calculator():
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
        try:
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
                monthly_payment = result
            elif unknown_param == "principal":
                st.success(f"Loan Amount: ${result:.2f}")
                principal = result
                monthly_payment = calculate_loan_parameter([principal, interest_rate, term, 0], "payment")
            elif unknown_param == "interest_rate":
                st.success(f"Annual Interest Rate: {result:.2f}%")
                interest_rate = result
                monthly_payment = calculate_loan_parameter([principal, interest_rate, term, 0], "payment")
            elif unknown_param == "term":
                if result > 1200:
                    st.error("Payment too small. Loan will never be fully paid off.")
                    monthly_payment = payment
                else:
                    st.success(f"Loan Term: {result:.0f} months ({result/12:.1f} years)")
                    term = result
                    monthly_payment = payment
            
            if unknown_param != "payment":
                st.info(f"Monthly Payment: ${monthly_payment:.2f}")
            
            if principal > 0 and interest_rate > 0 and term > 0 and monthly_payment > 0:
                total_payment = monthly_payment * term
                total_interest = total_payment - principal
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Payment", f"${total_payment:.2f}")
                with col2:
                    st.metric("Total Interest", f"${total_interest:.2f}")
                st.subheader("Loan Amortization")
                monthly_rate = interest_rate / (12 * 100)
                
                remaining_balance = np.zeros(int(term) + 1)
                interest_paid = np.zeros(int(term))
                principal_paid = np.zeros(int(term))
                
                remaining_balance[0] = principal
                
                for i in range(int(term)):
                    interest_paid[i] = remaining_balance[i] * monthly_rate
                    principal_paid[i] = monthly_payment - interest_paid[i]
                    remaining_balance[i + 1] = remaining_balance[i] - principal_paid[i]
                    if remaining_balance[i + 1] < 0:
                        remaining_balance[i + 1] = 0
                
                fig, ax = plt.subplots(figsize=(10, 6))
                months = np.arange(term + 1)
                ax.plot(months, remaining_balance, 'b-', label='Remaining Balance')
                ax.set_xlabel("Month")
                ax.set_ylabel("Amount ($)")
                ax.set_title("Loan Amortization Schedule")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                
                yearly_data = {
                    'Year': [f"Year {i+1}" for i in range(int(np.ceil(term/12)))],
                    'Principal Paid': [sum(principal_paid[i*12:min((i+1)*12, len(principal_paid))]) for i in range(int(np.ceil(term/12)))],
                    'Interest Paid': [sum(interest_paid[i*12:min((i+1)*12, len(interest_paid))]) for i in range(int(np.ceil(term/12)))]
                }
                yearly_df = pd.DataFrame(yearly_data)
                yearly_df['Principal Paid'] = yearly_df['Principal Paid'].map('${:,.2f}'.format)
                yearly_df['Interest Paid'] = yearly_df['Interest Paid'].map('${:,.2f}'.format)
                st.dataframe(yearly_df)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def render_investment_roi():
    st.header("Investment ROI Projection")
    st.write("Project investment returns using polynomial trend fitting")
    
    method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], horizontal=True)
    
    x, y = None, None
    
    if method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV with investment data", type="csv")
        if uploaded_file is not None:
            try:
                df = load_csv(uploaded_file)
                st.dataframe(df.head())
                
                x_col = st.selectbox("Time Period Column", df.columns)
                y_col = st.selectbox("Value/Return Column", df.columns)
                
                x = np.arange(len(df[x_col]))
                y = df[y_col].values
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
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
    
    if x is not None and y is not None and len(x) > 1:
        max_degree = min(5, len(x) - 1)
        degree = st.slider("Trend Complexity", 1, max_degree, min(2, max_degree), 
                          help="Higher values create more complex projections")
        projection_periods = st.slider("Projection Periods", 1, 10, 3)
        
        try:
            coeffs = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coeffs)
            
            future_x = np.array([max(x) + i + 1 for i in range(projection_periods)])
            future_y = polynomial(future_x)
            
            full_x = np.append(x, future_x)
            full_y = np.append(y, polynomial(future_x))
            
            initial_investment = y[0]
            final_value = future_y[-1]
            total_roi = (final_value - initial_investment) / initial_investment * 100
            years = (len(full_x) - 1) / 12 if max(x) <= 12 else len(full_x) - 1  
            annual_roi = ((final_value / initial_investment) ** (1 / years) - 1) * 100
            
            projection_df = pd.DataFrame({
                "Period": future_x,
                "Projected Value": future_y.round(2)
            })
            st.subheader("Investment Projection")
            st.dataframe(projection_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, color='blue', label='Historical Values')
            ax.scatter(future_x, future_y, color='red', label='Projected Values')
            
            smooth_x = np.linspace(min(x), max(future_x), 100)
            smooth_y = polynomial(smooth_x)
            ax.plot(smooth_x, smooth_y, 'g--', label='Trend Line')
            
            ax.set_xlabel("Period")
            ax.set_ylabel("Value ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for i, period in enumerate([min(x), max(x), max(future_x)]):
                idx = np.where(full_x == period)[0][0]
                value = full_y[idx]
                ax.annotate(f"${value:,.2f}", 
                           (period, value),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
            
            st.pyplot(fig)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total ROI", f"{total_roi:.2f}%")
            with col2:
                st.metric("Annualized ROI", f"{annual_roi:.2f}%")
            with col3:
                st.metric("CAGR", f"{annual_roi:.2f}%", 
                         help="Compound Annual Growth Rate")
            
            if len(x) > 2:
                predicted_historical = polynomial(x)
                mse = np.mean((y - predicted_historical)**2)
                rmse = np.sqrt(mse)
                st.metric("Volatility (RMSE)", f"${rmse:.2f}", 
                         help="Root Mean Square Error - measures prediction accuracy")
                
        except Exception as e:
            st.error(f"Error in projection: {str(e)}")

def render_bond_valuation():
    st.header("Bond Valuation & Yield Curve")
    st.write("Analyze bond prices and interpolate yields across different maturities")
    
    method = st.radio("Input Method", ["Upload CSV", "Use Sample Data"], key="bond_input", horizontal=True)
    
    maturities, yields = None, None
    
    if method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV with bond data", type="csv", key="bond_upload")
        if uploaded_file is not None:
            try:
                df = load_csv(uploaded_file)
                st.dataframe(df.head())
                
                maturity_col = st.selectbox("Maturity Column (years)", df.columns)
                yield_col = st.selectbox("Yield Column (%)", df.columns)
                
                maturities = df[maturity_col].values
                yields = df[yield_col].values
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
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
                
                idx = np.argsort(maturities)
                maturities = maturities[idx]
                yields = yields[idx]
                
                data_df = pd.DataFrame({"Maturity (Years)": maturities, "Yield (%)": yields})
                st.dataframe(data_df)
            except:
                st.error("Invalid data format. Please use 'maturity_years,yield_percent' on each line.")
    
    if maturities is not None and yields is not None and len(maturities) > 1:
        st.subheader("Bond Price Calculator")
        col1, col2, col3 = st.columns(3)
        with col1:
            face_value = st.number_input("Face Value ($)", min_value=100.0, value=1000.0, step=100.0)
        with col2:
            coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, value=4.0, step=0.25)
        with col3:
            maturity = st.number_input("Time to Maturity (years)", 
                                      min_value=float(min(maturities)), 
                                      max_value=float(max(maturities)),
                                      value=min(5.0, max(maturities)))
        
        try:
            market_yield, bond_price, tck = process_bond_data(maturities, yields, maturity, face_value, coupon_rate)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Yield", f"{market_yield:.2f}%")
            with col2:
                st.metric("Bond Price", f"${bond_price:.2f}")
            with col3:
                coupon_payment = face_value * (coupon_rate / 100) / 2
                periods = int(maturity * 2)
                discount_rate = market_yield / 100 / 2
                
                t_values = np.arange(1, periods + 1) / 2
                cf_values = np.ones(periods) * coupon_payment
                cf_values[-1] += face_value
                pv_factors = 1 / (1 + discount_rate) ** np.arange(1, periods + 1)
                pvs = cf_values * pv_factors
                
                duration = sum(t_values * pvs) / bond_price
                mod_duration = duration / (1 + discount_rate)
                
                st.metric("Modified Duration", f"{mod_duration:.2f} years", 
                         help="Price sensitivity to yield changes")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            dense_maturities = np.linspace(min(maturities), max(maturities), 100)
            dense_yields = interpolate.splev(dense_maturities, tck, der=0)
            
            ax.scatter(maturities, yields, color='blue', s=100, label='Market Yields')
            ax.plot(dense_maturities, dense_yields, 'r-', label='Yield Curve')
            ax.axvline(x=maturity, color='green', linestyle='--', label=f'Selected Maturity ({maturity} yrs)')
            ax.axhline(y=market_yield, color='green', linestyle='--')
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Yield (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("Yield Curve Analysis")
            
            short_yield = interpolate.splev(2, tck, der=0)
            long_yield = interpolate.splev(10, tck, der=0)
            mid_yield = interpolate.splev(5, tck, der=0)

            slope = long_yield - short_yield
            curvature = short_yield + long_yield - 2 * mid_yield
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Yield Curve Slope", f"{slope:.2f}%", 
                         "Normal" if slope > 0 else "Inverted (Recession Signal)")
            with col2:
                st.metric("Yield Curve Curvature", f"{curvature:.2f}%",
                         help="Positive: Humped curve, Negative: U-shaped curve")
            
            st.subheader("Yield Sensitivity Analysis")
            
            yield_changes = [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]
            prices = []
            
            for change in yield_changes:
                new_yield = market_yield + change
                new_discount_rate = new_yield / 100 / 2
                
                new_pv_coupons = coupon_payment * (1 - 1 / (1 + new_discount_rate) ** periods) / new_discount_rate if new_discount_rate > 0 else coupon_payment * periods
                new_pv_principal = face_value / (1 + new_discount_rate) ** periods
                new_price = new_pv_coupons + new_pv_principal
                prices.append(new_price)
            
            sensitivity_df = pd.DataFrame({
                "Yield Change": [f"{change:+.2f}%" for change in yield_changes],
                "New Yield": [f"{market_yield + change:.2f}%" for change in yield_changes],
                "Bond Price": [f"${price:.2f}" for price in prices],
                "Price Change": [f"{(price - bond_price):.2f}" for price in prices],
                "Change %": [f"{(price - bond_price) / bond_price * 100:.2f}%" for price in prices]
            })
            
            st.dataframe(sensitivity_df)
            
        except Exception as e:
            st.error(f"Error in bond calculations: {str(e)}")

def render_retirement_planning():
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
    
    st.subheader("Analysis Options")
    col1, col2 = st.columns(2)
    with col1:
        show_inflation_adjusted = st.checkbox("Show Inflation-Adjusted Values", value=True)
    with col2:
        contrib_increase_rate = st.slider("Annual Contribution Increase (%)", 0.0, 10.0, 0.0, 0.5,
                                         help="Increase monthly contributions annually with this rate")
    
    years_to_retirement = retirement_age - current_age
    months_to_retirement = years_to_retirement * 12
    
    real_return_rate = (1 + annual_return/100) / (1 + inflation_rate/100) - 1
    monthly_real_return = (1 + real_return_rate) ** (1/12) - 1
    
    nominal_monthly_return = (1 + annual_return/100) ** (1/12) - 1
    
    try:
        ages = np.arange(current_age, retirement_age + 1)
        nominal_savings = np.zeros(len(ages))
        real_savings = np.zeros(len(ages))
        nominal_savings[0] = initial_savings
        real_savings[0] = initial_savings
        
        contributions = np.zeros(len(ages))
        
        current_monthly = monthly_contribution
        
        for i in range(1, len(ages)):
            if i > 0:
                current_monthly *= (1 + contrib_increase_rate/100)
            
            annual_contribution = current_monthly * 12
            contributions[i-1] = annual_contribution
            
            nominal_savings[i] = nominal_savings[i-1] * (1 + annual_return/100) + annual_contribution
            real_savings[i] = real_savings[i-1] * (1 + real_return_rate) + annual_contribution / (1 + inflation_rate/100) ** (i)
        
        withdrawal_rate = 4.0 
        nominal_retirement_savings = nominal_savings[-1]
        real_retirement_savings = real_savings[-1]
        
        annual_nominal_income = nominal_retirement_savings * (withdrawal_rate / 100)
        monthly_nominal_income = annual_nominal_income / 12
        
        annual_real_income = real_retirement_savings * (withdrawal_rate / 100)
        monthly_real_income = annual_real_income / 12
        
        st.subheader("Retirement Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Nominal at Retirement", f"${nominal_retirement_savings:,.2f}")
            st.metric("Annual Nominal Income", f"${annual_nominal_income:,.2f}")
            st.metric("Monthly Nominal Income", f"${monthly_nominal_income:,.2f}")
        with col2:
            if show_inflation_adjusted:
                st.metric("Total Real at Retirement", f"${real_retirement_savings:,.2f}", 
                         f"{real_retirement_savings/nominal_retirement_savings*100:.1f}% of nominal")
                st.metric("Annual Real Income", f"${annual_real_income:,.2f}")
                st.metric("Monthly Real Income", f"${monthly_real_income:,.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(ages, nominal_savings, 'b-', marker='o', alpha=0.7, label='Nominal Savings')
        if show_inflation_adjusted:
            ax.plot(ages, real_savings, 'r-', marker='x', alpha=0.7, label='Inflation-Adjusted')
        
        ax.set_xlabel("Age")
        ax.set_ylabel("Savings ($)")
        ax.set_title("Retirement Savings Growth")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        annotation_indices = [0, len(ages)//2, -1]
        for i in annotation_indices:
            ax.annotate(f"${nominal_savings[i]:,.0f}", 
                       (ages[i], nominal_savings[i]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
            
            if show_inflation_adjusted:
                ax.annotate(f"${real_savings[i]:,.0f}", 
                           (ages[i], real_savings[i]),
                           textcoords="offset points",
                           xytext=(0, -15),
                           ha='center')
        
        st.pyplot(fig)
        
        # Анализ вклада
        st.subheader("Contribution Analysis")
        
        total_contributions = np.sum(contributions) + initial_savings
        investment_growth = nominal_savings[-1] - total_contributions
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Contributions", f"${total_contributions:,.2f}")
        with col2:
            st.metric("Investment Growth", f"${investment_growth:,.2f}", 
                     f"{investment_growth/total_contributions*100:.1f}% of contributions")
        
        additional_monthly = 100
        additional_savings = 0
        
        last_contrib = monthly_contribution
        for i in range(years_to_retirement):
            current_additional = additional_monthly * (1 + contrib_increase_rate/100) ** i
            future_value = current_additional * 12 * (1 + annual_return/100) ** (years_to_retirement - i)
            additional_savings += future_value
        
        if current_age >= 23:
            extra_years = 5
            delayed_savings = 0
            for i in range(years_to_retirement - extra_years):
                delayed_contrib = monthly_contribution * (1 + contrib_increase_rate/100) ** i
                future_value = delayed_contrib * 12 * (1 + annual_return/100) ** (years_to_retirement - extra_years - i)
                delayed_savings += future_value
            
            start_early_benefit = nominal_savings[-1] - delayed_savings
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bar_data = [nominal_savings[-1], delayed_savings]
            labels = ["Start Now", f"Delay {extra_years} Years"]
            ax.bar(labels, bar_data, color=['green', 'red'])
            
            for i, v in enumerate(bar_data):
                ax.text(i, v * 1.01, f"${v:,.0f}", ha='center')
            
            ax.set_ylabel("Retirement Savings ($)")
            ax.set_title(f"Impact of Delaying Retirement Savings by {extra_years} Years")
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("With Immediate Start", f"${nominal_savings[-1]:,.2f}")
            with col2:
                st.metric("With Delayed Start", f"${delayed_savings:,.2f}", 
                         f"-{start_early_benefit:,.2f}", delta_color="inverse")
        
        st.subheader("Impact of Additional Contributions")
        st.write(f"Adding ${additional_monthly} more per month would increase your retirement savings by: ${additional_savings:,.2f}")
        
        contribution_rates = [monthly_contribution * 0.5, 
                             monthly_contribution, 
                             monthly_contribution * 1.5, 
                             monthly_contribution * 2.0]
        
        final_amounts = []
        
        for rate in contribution_rates:
            savings = initial_savings
            monthly = rate
            
            for i in range(years_to_retirement):
                annual_contrib = monthly * 12 * (1 + contrib_increase_rate/100) ** i
                savings = savings * (1 + annual_return/100) + annual_contrib
            
            final_amounts.append(savings)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x_labels = [f"${r:.0f}/mo" for r in contribution_rates]
        ax.bar(x_labels, final_amounts, color='blue', alpha=0.7)
        
        for i, v in enumerate(final_amounts):
            ax.text(i, v * 1.01, f"${v:,.0f}", ha='center')
        
        ax.set_ylabel("Retirement Savings ($)")
        ax.set_title("Impact of Different Monthly Contribution Rates")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error in retirement calculations: {str(e)}")

if __name__ == "__main__":
    main()
