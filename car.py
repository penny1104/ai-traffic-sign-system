import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import RPi.GPIO as GPIO
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
import time
import os
import sys

# ---------------------------
# 類別名稱
# ---------------------------
class_names = ['green', 'left', 'none', 'red', 'right', 'slow', 'stop']

# ---------------------------
# TinyCNN 定義
# ---------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )       

    def forward(self,x):
        return self.classifier(self.features(x))

# ---------------------------
# 載入模型
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN(num_classes=7).to(device)
model_path = os.path.join(os.path.dirname(__file__), "final.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------------------
# 影像預處理
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def predict_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

# ---------------------------
# GPIO 與馬達設定
# ---------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# 馬達 A (左輪)
ENA, IN1, IN2 = 19, 20, 21
# 馬達 B (右輪)
ENB, IN3, IN4 = 18, 23, 24
# 超音波感測器
TRIG, ECHO = 12, 13

for pin in [ENA, IN1, IN2, ENB, IN3, IN4, TRIG]:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# ---------------------------
# 伺服馬達設定
# ---------------------------
factory = PiGPIOFactory()
time.sleep(0.5)
STRAIGHT_OFFSET = 5

servo = AngularServo(
    17,
    min_angle=-70, max_angle=70,
    min_pulse_width=0.0008, max_pulse_width=0.0022,
    pin_factory=factory
)

def center_steering():
    servo.angle = STRAIGHT_OFFSET
    time.sleep(0.3)

def steer(deg):
    deg = max(-60, min(60, deg))
    servo.angle = deg
    time.sleep(0.2)

# ---------------------------
# 馬達控制動作
# ---------------------------
def both_forward(speed):
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def both_stop():
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.LOW)

def backward(speed=30):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def slow_down(speed=25):
    both_forward(speed)
    time.sleep(3)
# ---------------------------
#轉向控制 + 持續時間
# ---------------------------
def turn_with_steering(speed, duration, direction='right'):
    if direction == 'right':
        print("右轉中...")
        steer(-45)
        both_forward(speed)
    else:
        print("左轉中...")
        steer(45)
    both_forward(speed)
    time.sleep(duration)
    both_stop()
    center_steering()
# ---------------------------
# 超音波測距
# ---------------------------
def get_distance(timeout=0.05):
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    timeout = pulse_start + 0.04
    while GPIO.input(ECHO) == 0 and time.time() < timeout:
        pulse_start = time.time()
    pulse_end = time.time()
    timeout = pulse_end + 0.04
    while GPIO.input(ECHO) == 1 and time.time() < timeout:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # cm
    return distance

# ---------------------------
# 主程式迴圈
# ---------------------------
cap = cv2.VideoCapture(0)

try:
    print("🚗自走車啟動")
    center_steering()
    while True:
        # 預設直走
        both_forward(speed=40)
        distance = get_distance()   

        ret, frame = cap.read()
        if not ret:
            continue

        # 距離偵測
        if distance < 40:
            print("障礙物偵測：停止 + 後退 + 右轉")
            both_stop()
            backward(speed=30)
            time.sleep(2)
            turn_with_steering(speed=35, duration=5, direction='right')
            continue
        cls = predict_image(frame)

        # 辨識結果判斷
        if distance <= 75:
            if cls == "none":
                pass
            elif cls == "red":
                both_stop()
                time.sleep(3)
            elif cls == "stop":
                both_stop()
                backward()
                time.sleep(2)
                turn_with_steering(speed=35, duration=3.3, direction='left')
            elif cls == "green":
                pass
            elif cls == "left":
                turn_with_steering(speed=35, duration=1.6, direction='left')
            elif cls == "right":
                turn_with_steering(speed=35, duration=1.6, direction='right')
            elif cls == "slow":
                slow_down(speed=25)
            print(f"Prediction: {cls}, Distance: {distance:.1f}cm")
            time.sleep(1)

except KeyboardInterrupt:
    print("程式中斷")

finally:
    both_stop()
    pwm_a.stop(); pwm_b.stop()
    try:
        servo.detach()
    except Exception:
        pass
    GPIO.cleanup()
    cap.release()
    print("✅程式安全退出")