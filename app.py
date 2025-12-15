import streamlit as st
import pandas as pd
import json
import torch
import google.generativeai as genai
import requests # 用於 Pollinations
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="易經智慧導航", page_icon="☯️", layout="centered")

# 自定義 CSS
st.markdown("""
<style>
    .stTextInput > label {font-size:110%; font-weight:bold;}
    .stTextArea > label {font-size:110%; font-weight:bold;}
    
    .hexagram-box {
        background-color: #FFF8E1; 
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #D97706;
        margin: 20px 0;
        font-family: "KaiTi", "BiauKai", serif;
        color: #2e2e2e;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 載入資料與模型 ---
@st.cache_resource
def load_resources():
    try:
        df = pd.read_json('yi_jing_data.json')
        df['search_text'] = df['modern_translation'] + " " + df['meaning_keywords']
    except Exception as e:
        st.error(f"找不到 yi_jing_data.json，請確認檔案位置。錯誤訊息: {e}")
        return None, None, None

    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    embeddings = model.encode(df['search_text'].tolist(), convert_to_tensor=True)
    
    return df, model, embeddings

df, embed_model, doc_embeddings = load_resources()

# --- 3. 側邊欄設定 ---
with st.sidebar:
    st.title("🔮 設定")
    
    st.subheader("1. 講給你聽 (文字推理)")
    groq_api_key = st.text_input("Groq API Key", type="password", help="gsk_開頭")
    st.markdown("[取得 Groq Key](https://console.groq.com/keys)")

    st.divider()

    st.subheader("2. 畫給你看 (圖像生成)")
    image_model_choice = st.radio(
        "選擇繪圖模型",
        [
            "Pollinations (免鑰匙/免費/推薦) 👍", 
            "Google Imagen 3 (高品質/需權限)", 
            "OpenAI DALL·E 3 (付費/高品質)"
        ]
    )

    if image_model_choice == "OpenAI DALL·E 3 (付費/高品質)":
        openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        google_api_key = None
        st.markdown("[取得 OpenAI Key](https://platform.openai.com/api-keys)")
        
    elif image_model_choice == "Google Imagen 3 (高品質/需權限)":
        google_api_key = st.text_input("Google API Key", type="password", help="AIza開頭", key="google_key")
        openai_api_key = None
        st.markdown("[取得 Google API Key](https://aistudio.google.com/app/apikey)")
        
    else: # Pollinations
        st.info("✨ 使用 Pollinations.ai 技術，完全免費，無需設定 API Key。")
        google_api_key = None
        openai_api_key = None

    st.divider()

    role = st.selectbox(
        "選擇 AI 解卦風格",
        ["智慧長者 (溫暖指引)", "嚴肅老師 (簡潔有力)", "心理諮商師 (同理分析)", "白話翻譯機 (直白易懂)"]
    )
    
    st.caption("Designed for I Ching AI Project")

# --- 4. 主介面邏輯 ---
st.title("☯️ 易經智慧對話機器人")
st.markdown("請在心中默念您的疑問，描述當下的處境，AI 將為您感應最適合的一卦。")

with st.form("query_form"):
    col1, col2 = st.columns(2)
    with col1:
        user_event = st.text_area("1. 發生了什麼事？請儘量描述細節與情境", height=400, placeholder="例如：剛換新工作，同事很難相處...")
    with col2:
        user_question = st.text_area("2. 想問什麼？儘量聚焦在自己可以改變的地方", height=400, placeholder="例如：我該離職還是繼續撐下去？")
    
    submitted = st.form_submit_button("🔍 易經哲學觀")

# --- 5. 執行運算 ---
if submitted:
    if not groq_api_key:
        st.warning("⚠️ 請先在左側側邊欄輸入 Groq API Key 才能運作喔！")
    elif not user_event or not user_question:
        st.warning("⚠️ 請完整輸入「事件」與「想問的問題」。")
    else:
        full_query = f"事件：{user_event}。疑問：{user_question}"
        
        with st.spinner('正在連結傳承千年的智慧，易經不是算命，是告訴你目前處境在哪個節點...'):
            # A. 向量搜尋
            query_embedding = embed_model.encode(full_query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_match_idx = torch.argmax(cos_scores).item()
            result = df.iloc[best_match_idx]
            
            hex_name = f"{result['hexagram']} {result['position']}"
            original_text = result['original_text']
            translation = result['modern_translation']
            
        st.success(f"卜得：【 {hex_name} 】")
        st.markdown(f"""
        <div class="hexagram-box">
            <b>📜 經文：</b>{original_text}<br>
            <b>📖 解釋：</b>{translation}
        </div>
        """, unsafe_allow_html=True)

        # B. LLM 解讀
        with st.spinner('智慧之書正在撰寫解籤...'):
            try:
                chat = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
                
                system_prompt = f"你現在是一位精通易經的{role}。請根據提供的卦象資訊，回答使用者的問題。"
                
                human_prompt = f"""
                使用者情境：{full_query}
                
                對應易經爻辭結果：
                卦名：{hex_name}
                原文：{original_text}
                白話意涵：{translation}
                
                請依照以下 Markdown 格式結構進行**詳盡且深入**的解讀（總字數請超過 600 字）：
                
                ### 1. 🔍 卦象深度解析
                - 這一部份至少 300 字。
                - 請務必將易經原文完整寫出，並解釋其意象（例如：山下有火、雷在天上等代表什麼自然現象？）。
                - 結合易經原文，分析為什麼這個卦象精準對應到了使用者的「{user_event}」情境？
                - 解析這一爻（{result['position']}）在整個卦中的位置意義（是吉是凶？是剛開始還是快結束？）。

                ### 2. 💡 局勢判讀與心理建設
                - 分析此刻「{user_question}」背後的心理狀態。
                - 點出目前局勢的潛在風險是什麼？
                - 有什麼看不見的機會點正在萌芽？

                ### 3. 🚀 具體行動指南 (Step-by-Step)
                - 請給出 3 到 5 個具體的執行步驟，不要只有空泛的建議。
                - 分別針對「短期（現在立刻做）」與「長期（未來一個月）」給予建議。

                ### 4. ⚠️ 提醒警語
                - 如果不聽勸告，最壞的情況會是如何？
                - 提醒使用者在心態上要避免的盲點（例如：避免急躁、避免貪心）。

                ### 5. 🌈 智慧結語
                (請在此處直接撰寫一段溫暖、充滿力量的祝福或定心丸，不需要列點，也不要重複此指令)

                ---
                (重要：請在回答的最後面，獨立一行，提供一段約 50-80 個單字的「英文」圖像生成提示詞 (Image Prompt)。這段提示詞要能視覺化呈現你上述「卦象深度解析」中的核心意境與氛圍，風格要求為：寫實風景畫結合後現代藝術與哲學感。請務必以 "IMAGE_PROMPT:" 開頭。)
                """
                
                messages = [("system", system_prompt), ("human", human_prompt)]
                
                # 呼叫 LLM
                ai_response = chat.invoke(messages)
                full_response_text = ai_response.content

                # 解析回應，分離圖片提示詞
                if "IMAGE_PROMPT:" in full_response_text:
                    parts = full_response_text.split("IMAGE_PROMPT:")
                    text_display = parts[0].strip()
                    image_prompt_en = parts[1].strip()
                else:
                    text_display = full_response_text
                    image_prompt_en = None

                # 顯示文字
                st.markdown("### 💡 智慧指引")
                st.write(text_display)
                st.divider()

                # C. 圖像生成邏輯 (三大門派)
                if image_prompt_en:
                    # 門派 1: Pollinations (免鑰匙/最穩)
                    if image_model_choice == "Pollinations (免鑰匙/免費/推薦) 👍":
                        with st.spinner('正在描繪心靈圖騰 (Pollinations)...'):
                            try:
                                # Pollinations 是透過 URL 直接生圖，無需 SDK
                                safe_prompt = image_prompt_en.replace(" ", "%20")
                                # 加入 seed 讓圖片更隨機
                                import random
                                seed = random.randint(1, 99999)
                                image_url = f"https://image.pollinations.ai/prompt/A%20philosophical%20Chinese%20ink%20painting%20{safe_prompt}?nologo=true&seed={seed}&width=1024&height=1024"
                                st.image(image_url, caption=f"【{hex_name}】心靈意境圖", use_column_width=True)
                            except Exception as e:
                                st.error(f"Pollinations 圖片生成失敗：{e}")

                    # 門派 2: Google Imagen 3
                    elif image_model_choice == "Google Imagen 3 (高品質/需權限)" and google_api_key:
                        with st.spinner('Imagen 正在描繪心靈圖騰...'):
                            try:
                                genai.configure(api_key=google_api_key)
                                imagen_model = genai.ImageGenerationModel("imagen-3.0-generate-001")
                                result = imagen_model.generate_images(
                                    prompt=f"A philosophical and artistic illustration style, Chinese ink painting aesthetic combined with abstract modern art. {image_prompt_en}",
                                    number_of_images=1,
                                )
                                st.image(result.images[0], caption=f"【{hex_name}】Google Imagen 3 意境圖", use_column_width=True)
                            except AttributeError:
                                st.error("❌ 你的 Google 套件版本過舊。請在終端機執行 `pip install -U google-generativeai` 更新。")
                            except Exception as e:
                                st.error(f"Google 圖片生成失敗：{e} (可能是帳號權限問題，請改用 Pollinations)")

                    # 門派 3: OpenAI DALL-E 3
                    elif image_model_choice == "OpenAI DALL·E 3 (付費/高品質)" and openai_api_key:
                        with st.spinner('DALL·E 正在描繪心靈圖騰...'):
                            try:
                                from openai import OpenAI
                                client = OpenAI(api_key=openai_api_key)
                                response = client.images.generate(
                                    model="dall-e-3",
                                    prompt=f"A philosophical and artistic illustration style, Chinese ink painting aesthetic combined with abstract modern art. {image_prompt_en}",
                                    size="1024x1024",
                                    quality="standard",
                                    n=1,
                                )
                                image_url = response.data[0].url
                                st.image(image_url, caption=f"【{hex_name}】DALL·E 3 意境圖", use_column_width=True)
                            except Exception as e:
                                st.error(f"DALL·E 圖片生成失敗：{e}")
                    
                    elif not openai_api_key and not google_api_key and "Pollinations" not in image_model_choice:
                        st.info("ℹ️ 請在左側輸入對應的 API Key，或者選擇 Pollinations 免鑰匙模式！")

            except Exception as e:
               st.error(f"連線錯誤：{e} (請檢查 API Key 是否正確)")
               # streamlit run app.py
