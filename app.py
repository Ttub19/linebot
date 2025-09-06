# -*- coding: utf-8 -*-

import os
import pickle
import tempfile
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

from flask import Flask, request, abort
from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
)

import google.generativeai as gemini
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

import os
from dotenv import load_dotenv
load_dotenv()

line_channel_access_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
line_channel_secret = os.environ.get("LINE_CHANNEL_SECRET")
gemini_api_key = os.environ.get('GEMINI_API_KEY')

if not line_channel_access_token or not line_channel_secret:
    print("WARNING: LINE environment variables not set. Using hardcoded values.")
    line_channel_access_token = 'z+JWq427fAW3tKluFvqiLusHQcwIoBbjAsukqA+FH2nExXD6xCQp3rds0XuD7igyplYl05Q3PdN2DKPswWhUic6d57iYX6x6bS93hQXz+HS0G1M/6ib0A5dF231oT9UBCCPuuXk4t09TqK6p8h5o3QdB04t89/1O/w1cDnyilFU='
    line_channel_secret = '27f5c410c83b1d542909fdc3e84a4588'

configuration = Configuration(access_token=line_channel_access_token)
handler = WebhookHandler(line_channel_secret)

# --- Gemini API Configuration ---
if not gemini_api_key:
    print("WARNING: GEMINI_API_KEY not set in environment variables. Using hardcoded value.")
    gemini_api_key = 'AIzaSyDQf5LwwytpclG4uJIjep36FfmzZocQT3A'

gemini.configure(api_key=gemini_api_key)
gemini_model = gemini.GenerativeModel('gemini-2.0-flash')
GEMINI_PROMPT = "จงทำตัวเป็นซามูไรที่เป็นคนเหนือพูดภาษาเหนือ"

# --- Image Classification Model Configuration ---
LABELS = {
    0: "แมวมีปุกบินไม่ได้",
    1: "หมาล่าอร่อยดี",
    2: "ยีราฟคอสั้นกินข้าว"
}
IMAGE_SIZE = (64, 64)

# Load the pre-trained SVM model for image prediction
img_model = None
try:
    with open('img_models.sav', 'rb') as f:
        img_model = pickle.load(f)
        print("Model 'img_models.sav' loaded successfully.")
except FileNotFoundError:
    print("ERROR: The 'img_models.sav' file was not found.")
    print("Please run 'create_model.py' first to create and save the model.")
except Exception as e:
    print(f"ERROR: Could not load 'img_models.sav'. Reason: {e}")

# --- Accuracy Calculation Function ---
def calculate_accuracy(model):
    """
    Calculates the accuracy of the loaded model using mock test data.
    Replace this with your actual data loading code.
    """
    if not model:
        print("ERROR: Cannot calculate accuracy. Model is not loaded.")
        return

    try:
        # --- Mock Test Data ---
        # NOTE: You MUST replace this with your actual test data
        # For this example, we'll create some random data.
        X_test = np.random.rand(100, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3)
        y_test = np.random.randint(0, 3, size=100)
        # --- End of Mock Data ---

        # Make predictions on the test data
        predictions = model.predict(X_test)
        
        # Calculate and print the accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.2f}")

    except Exception as e:
        print(f"ERROR: An exception occurred during accuracy calculation: {e}")

# Call the function to calculate accuracy after the model is loaded
if img_model:
    calculate_accuracy(img_model)

# --- Prediction Function ---
def predict_category(image_path, model):
    if not model:
        return "ไปโหลดโมดเลก่อนเด้ออ้ายบ่าว"

    try:
        img = Image.open(image_path)
        img = img.resize(IMAGE_SIZE)
        img = img.convert('RGB')
        
        test_data = np.array(img).flatten()
        
        predicted_label_index = model.predict([test_data])[0]
        predicted_category = LABELS.get(predicted_label_index, "ตัวไอ่ไหร่นิไม่รูัจักเน้ออพี่บ่าวเห้อะ")
        return predicted_category
    except FileNotFoundError:
        return "ไฟล์รูปภาพไม่พบ"
    except Exception as e:
        print(f"CRITICAL ERROR: An exception occurred during prediction: {e}")
        return "เกิดข้อผิดพลาดในการดูลูกแก้ว"

# --- LINE Bot Webhook Handler ---
@app.route('/webhook', methods=['POST'])
def webhook():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    
    return 'OK'

### Message Event Handlers

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        user_text = event.message.text
        full_prompt = f"{GEMINI_PROMPT}\n\n{user_text}"
        
        reply_text = "ขออภัยครับ ขณะนี้ไม่สามารถประมวลผลคำขอได้ กรุณาลองอีกครั้งในภายหลังถ้าทนไม่ไหวมึงก็ไม่ต้องมาทำอะไรทั้งนั้น"
        
        try:
            response = gemini_model.generate_content(full_prompt)
            if response and response.text:
                reply_text = response.text
            else:
                app.logger.warning("Gemini API returned an empty response.")
        except Exception as e:
            app.logger.error(f"Error while calling Gemini API for text message: {e}")
        finally:
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_blob_api = MessagingApiBlob(api_client)
        reply_text = "ขออภัยครับมีข้อขัดขาจนล้มโอ้ยยเจ็บ"
        temp_file_path = None
        
        try:
            message_content = line_blob_api.get_message_content(event.message.id)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tf:
                tf.write(message_content)
                temp_file_path = tf.name
            
            predicted_category = predict_category(temp_file_path, img_model)
            reply_text = f"รูปนี้คือ: {predicted_category}"

        except Exception as e:
            app.logger.error(f"Error handling image message: {e}")
            reply_text = "ขออภัยครับมีข้อขัดแย้งนิดมากหน่อย"
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )

# --- Main execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)