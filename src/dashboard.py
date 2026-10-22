import streamlit as st
from predict import score_transaction

st.set_page_config(page_title='Fraud Risk Scoring', page_icon='shield', layout='wide')
st.title('Transaction Fraud Risk Scoring')
st.caption('ML demo using behavioral, device, merchant and transaction signals.')

with st.form('transaction_form'):
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input('Amount', min_value=1.0, value=250.0)
    hour = c2.slider('Hour', 0, 23, 14)
    distance = c3.number_input('Distance (km)', 0.0, 300.0, 15.0)
    age = c1.number_input('Account age (days)', 0, 5000, 240)
    count = c2.number_input('Transactions / 24h', 0, 50, 3)
    velocity = c3.number_input('Velocity score', 0.0, 50.0, 2.0)
    trust = c1.slider('Device trust', 0.0, 1.0, 0.75)
    merchant = c2.slider('Merchant risk', 0.0, 1.0, 0.15)
    international = c3.checkbox('International')
    new_device = c1.checkbox('New device')
    weekend = c2.checkbox('Weekend')
    failed = c3.number_input('Failed logins', 0, 20, 0)
    chargeback = c1.checkbox('Chargeback history')
    submitted = st.form_submit_button('Score transaction')

if submitted:
    result = score_transaction({
        'amount': amount, 'hour': hour, 'distance_km': distance,
        'account_age_days': age, 'tx_count_24h': count,
        'velocity_score': velocity, 'device_trust': trust,
        'merchant_risk': merchant, 'international': int(international),
        'new_device': int(new_device), 'weekend': int(weekend),
        'failed_logins': failed, 'chargeback_history': int(chargeback)
    })
    st.metric('Fraud probability', f"{result['fraud_probability']:.1%}")
    st.subheader(f"Risk level: {result['risk_level']}")
