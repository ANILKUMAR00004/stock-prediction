import streamlit as st
import random
rotate = """
    <style>
    .money {
        position: fixed;
        top: -50px;
        font-size: 2rem;
        z-index: 9999;
        animation: fall linear forwards;
    }
    @keyframes fall {
        to {
            transform: translateY(105vh) rotate(360deg);
        }
    }
    </style>
    """
def money_showers():
    st.markdown(rotate, unsafe_allow_html=True)    
    import random
    html_code = ""
    emojis = ["💵", "💰", "💸", "🤑"]
    
    for _ in range(40):
        emoji = random.choice(emojis)
        left = random.randint(0, 95)  
        delay = random.uniform(0, 3)   
        duration = random.uniform(2, 5) 
        
        html_code += f'''
        <div class="money" style="left: {left}vw; animation-delay: {delay}s; animation-duration: {duration}s;">
            {emoji}
        </div>
        '''
    
    st.markdown(html_code, unsafe_allow_html=True)
def money_shower_loss():
    st.markdown(rotate, unsafe_allow_html=True)
    
    html_code = ""
    emojis = ["📉", "💸", "🚨", "🛑", "🔻"]
    for _ in range(50):
        emoji = random.choice(emojis)
        left = random.randint(0, 95)
        delay = random.uniform(0, 2)
        duration = random.uniform(1.5, 3.5) 
        html_code += f'''<div class="money" style="left: {left}vw;animation-delay: {delay}s;animation-duration:{duration}s;">
        {emoji}</div>'''
    st.markdown(html_code, unsafe_allow_html=True)
