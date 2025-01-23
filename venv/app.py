from flask import Flask, render_template, redirect, url_for, Response, jsonify, session, flash, request
import cv2
import numpy as np
import face_recognition
import time
from flask_socketio import SocketIO, send
import torch
import pandas as pd
import requests
import git
import threading
from pathlib import Path
import pathlib
import mysql.connector
import os
from sklearn import svm
import math

# from yolov5.models.experimental import attempt_load
# import yolov5

app = Flask(__name__)
app.secret_key = 'your_secret_key'

enable_switch = False
model = None
bmodel = None
bmodelcoat = None

model_path='weights/best.pt'

# model1 = None
# model2 = None
ch_switch1 = False
ch_switch0 = False
b_switchCoat = False
b_switchShoes = False
entry_name = None
detection_status = {
    'gloves': False,
    'goggles': False,
    'labcoat': False
}
capture_flag = False
capture_status = None
return_flag = False

user_data = {}

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def load_yolov5_model(weights_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)


def reset_variables():
    global enable_switch, ch_switch1, ch_switch0, b_switchCoat, b_switchShoes, detection_status, entry_name, bmodel, bmodelcoat
    enable_switch = False
    ch_switch1 = False
    ch_switch0 = False
    b_switchCoat = False
    b_switchShoes = False
    entry_name = None
    detection_status = {
        'gloves': False,
        'goggles': False,
        'labcoat': False
    }
    bmodel=None
    bmodelcoat=None


@app.route('/')
def index():
    reset_variables()
    if 'entry_name' in session and session['entry_name'] is not None:
        session.pop('entry_name')
    if 'enable_switch' in session and session['enable_switch'] is not None:
        session.pop('enable_switch')
    return render_template('main.html')


#db 연결
def db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0000",
        database="mydb",
        port="3306"
    )
    return conn


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM admin WHERE admin_id = %s AND admin_pw = %s', (username, password,))
        account = cursor.fetchone()

        if account:
            session['username'] = account[1]
            return redirect(url_for('index'))
        else:
            msg = '등록된 관리자가 아닙니다! 다시입력해보세요'
    return render_template('login.html', error=msg)


@app.route('/logout')
def logout():
    session.pop('username', None)  # 세션에서 사용자 정보 삭제
    return redirect(url_for('index'))


@app.route('/admin')
def admin():
    if 'username' in session:
        return render_template('adminpage.html')
    else:
        flash('권한이 없습니다. 로그인하세요')
        return redirect(url_for('login'))


@app.route('/start_page', methods=['GET', 'POST'])
def start_page():
    global enable_switch
    # enable_switch = False
    print("start_page", enable_switch)
    return render_template('face_rec.html', enable_switch=enable_switch)


@app.route('/check_enable_switch')
def check_enable_switch():
    global enable_switch
    return jsonify({'enable_switch': enable_switch})


#사용자 등록
#사용자 정보 입력
@app.route('/check_id')
def check_id():
    user_id = request.args.get('userId')
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})


@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    global user_data
    if 'username' in session:
        if request.method == 'POST':
            user_id = request.form['userId']
            user_name = request.form['userName']
            user_type = request.form['user_type']

            user_data = {
                'id': user_id,
                'name': user_name,
                'user_type': user_type
            }
            print(user_data)

            return redirect(url_for('photo_shoot'))
        return render_template('add_user.html')
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))


@app.route('/photo_shoot')
def photo_shoot():
    if 'username' in session:
        return render_template('photo_shoot.html', user_data=user_data)
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))


@app.route('/run_rec')
def run_rec():
    if 'username' in session:
        return render_template('run_rec.html')
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))


@app.route('/train_faces', methods=['POST'])
def train_faces():
    print("train_faces 시작")
    success = face_training()
    print("train_faces 끝")
    return jsonify({"success": success})


@app.route('/user_info')
def user_info():
    if 'username' in session:
        page = request.args.get('page', 1, type=int)
        items_per_page = 12

        conn = db_connection()
        cursor = conn.cursor()

        # 전체 행 수 가져오기
        cursor.execute("SELECT COUNT(*) FROM users")
        total_rows = cursor.fetchone()[0]
        total_pages = math.ceil(total_rows / items_per_page)

        # 현재 페이지에 보여줄 데이터
        offset = (page - 1) * items_per_page
        cursor.execute("SELECT * FROM users LIMIT %s OFFSET %s", (items_per_page, offset))
        data = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template('user_info.html', data_list=data, current_page=page, total_pages=total_pages)
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))


@app.route('/access_rec')
def access_rec():
    if 'username' in session:
        page = request.args.get('page', 1, type=int)
        items_per_page = 12

        conn = db_connection()
        cursor = conn.cursor()

        # 전체 행 수 가져오기
        cursor.execute("SELECT COUNT(*) FROM access")
        total_rows = cursor.fetchone()[0]
        total_pages = math.ceil(total_rows / items_per_page)

        offset = (page - 1) * items_per_page
        cursor.execute("SELECT access.*, users.name FROM access JOIN users ON access.users_id = users.id "
                       "ORDER BY access.time DESC LIMIT %s OFFSET %s", (items_per_page, offset))
        data = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template('access_rec.html', data_list=data, current_page=page, total_pages=total_pages)
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))


@app.route('/lab_page')
def lab_page():
    global entry_name, enable_switch
    session['entry_name'] = entry_name
    session['enable_switch'] = enable_switch
    return render_template('lab_opt.html')


@app.route('/basic_page')
def basic_page():
    global bmodel, model_path
    global b_switchCoat, b_switchShoes
    # bmodel = load_yolov5_model('weights/0401m3.pt')
    bmodel = load_yolov5_model(model_path)
    return render_template('basic.html', b_switchCoat=b_switchCoat, b_switchShoes=b_switchShoes)


@app.route('/chemical_page')
def chemical_page():
    global model
    global ch_switch0, ch_switch1
    model = load_yolov5_model('weights/best.pt')
    return render_template('chemical.html', ch_switch0=ch_switch0, ch_switch1=ch_switch1)


#사용자 등록 사진 촬영
def capture_images():
    global user_data
    global capture_flag, capture_status, return_flag
    # 사용자로부터 ID 입력 받기
    # user_id = input("Enter user ID: ")

    # 웹캠 초기화
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        capture_status = 'fail'
        return_flag = True
        return

    # 얼굴 검출을 위한 Haar Cascade 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    capture_flag = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image1")
            capture_status = 'fail'
            return_flag = True
            break

        # 얼굴 검출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0 and capture_flag:
            text = f"Capture Count: {count + 1}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            user_id = user_data['id']
            save_path = "C:\\SavePicture"
            user_folder = os.path.join(save_path, user_id)
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            face_filename = os.path.join(user_folder, f"{user_id}_{count + 1}.jpg")
            cv2.imwrite(face_filename, frame)
            count += 1

            if count >= 10:
                capture_flag = False
                capture_status = 'success'
                return_flag = True
                break

            time.sleep(0.5)

        # 화면에 녹색 프레임 표시
        frame_with_rect = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret2, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/capture_feed')
def capture_feed():
    return Response(capture_images(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_image', methods=['POST'])
def capture_image():
    global user_data
    global capture_flag, return_flag, capture_status
    capture_flag = True
    return_flag = False

    while not return_flag:
        time.sleep(0.1)

    if capture_status == 'success':
        # 데이터베이스에 사용자 데이터 저장
        conn = db_connection()
        cursor = conn.cursor()
        try:
            sql = "INSERT INTO users (id, name, user_type) VALUES (%s, %s, %s)"
            cursor.execute(sql, (user_data['id'], user_data['name'], user_data['user_type']))
            conn.commit()
            print("사용자 데이터를 데이터베이스에 추가했습니다.")
        except Exception as e:
            conn.rollback()
            print("사용자 데이터를 데이터베이스에 추가하는 도중 오류가 발생했습니다:", str(e))
            capture_status = 'fail'
        finally:
            cursor.close()
            conn.close()
    
    return jsonify({'status': capture_status})


def face_training():
    train_dir = os.listdir('C:\\SavePicture\\')

    known_face_encodings = []
    known_face_names = []

    for person in train_dir:
        pix = os.listdir('C:\\SavePicture\\' + person)

        # 각 사람의 이미지
        for person_img in pix:
            face = face_recognition.load_image_file('C:\\SavePicture\\' + person + "\\" + person_img)
            face_encodings = face_recognition.face_encodings(face)
            if len(face_encodings) > 0:  # 얼굴 인코딩 있으면
                face_enc = face_encodings[0]
                known_face_encodings.append(face_enc)
                known_face_names.append(person)
            else:
                continue

    # SVC 분류기 생성, 훈련
    clf = svm.SVC(gamma='scale')
    clf.fit(known_face_encodings, known_face_names)

    # 훈련된 모델 저장
    np.save('known_face_encodings_1.npy', known_face_encodings)
    np.save('known_face_names_1.npy', known_face_names)

    return True


#얼굴 인식
def gen_frames():
    # 훈련된 모델 불러오기
    known_face_encodings = np.load('known_face_encodings_1.npy')
    known_face_names = np.load('known_face_names_1.npy')
    global entry_name

    # 웹캠
    video_capture = cv2.VideoCapture(0)

    start_time = None  # 시작 시간을 저장할 변수 초기화
    global enable_switch

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                min_value = min(distances)
                name = "Unknown"
                if min_value < 0.4:
                    index = np.argmin(distances)
                    name = known_face_names[index]
                    if name != "Unknown" and start_time is None:
                        start_time = time.time()

                face_names.append(name)

            if (all(name != face_names[0] for name in face_names)):
                start_time = None

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name == "Unknown":
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if all(name == face_names[0] for name in face_names) and start_time is not None and time.time() - start_time >= 5:
                cv2.putText(frame, "Success!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            if (all(name == face_names[0] for name in face_names)
                    and start_time is not None and time.time() - start_time >= 6):
                print("face:", name)
                enable_switch = True
                print("gen_frame", enable_switch)
                entry_name = name
                print("entry_name:",entry_name)
                break

            if all(name == "Unknown" for name in face_names) or not face_names:
                start_time = None

            # 화면 하단에 텍스트 추가
            cv2.putText(frame, "face recognition", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#화학 실험실 웹캠
def gen_chemical1():
    global ch_switch1, ch_switch0
    global model
    global detection_status
    model1 = model
    # model = torch.hub.load('ultralytics/yolov5', 'custom',
    #                        path='weights/best.pt', force_reload=True)

    # 각 객체별 신뢰도 임계값 설정
    confidence_thresholds = {
        'gloves': 0.9,
        'goggles': 0.7,
        'labcoat': 0.8
    }

    # 객체별 바운딩 박스 색상 설정
    colors = {
        'gloves': (255, 0, 0),  # Red
        'goggles': (0, 255, 0),  # Green
        'labcoat': (0, 0, 255)  # Blue
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(0)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = model1(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")
                    detection_status[obj_name] = True

            # 모든 지정된 객체들이 3초 이상 연속으로 탐지되었는지 확인
            if all(continuous_detection[obj_name] for obj_name in confidence_thresholds):
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                ch_switch1 = True
                cv2.waitKey(3000)  # 3초간 대기
                if ch_switch0 and ch_switch1:
                    print('문이 열렸습니다.')
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


@app.route('/chemical_feed1')
def chemical_feed1():
    return Response(gen_chemical1(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_detection_status')
def check_detection_status():
    global detection_status
    return jsonify(detection_status)


def gen_chemical0():
    global ch_switch0, ch_switch1
    global model
    model2 =model
    # model = torch.hub.load('ultralytics/yolov5', 'custom',
    #                        path='weights/best.pt', force_reload=True)

    # 각 객체별 신뢰도 임계값 설정
    confidence_thresholds = {
        'shoes': 0.85,
        'No-shoes': 0.85
    }

    # 객체별 바운딩 박스 색상 설정
    colors = {
        'shoes': (255, 255, 0),  # Yellow
        'No-shoes': (255, 105, 180)  # Pink
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(1)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = model2(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")

            # 모든 지정된 객체들(No-shoes 제외)이 3초 이상 연속으로 탐지되었는지 확인
            if continuous_detection['shoes']:
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                ch_switch0 = True
                cv2.waitKey(3000)  # 3초간 대기
                if ch_switch1 and ch_switch0:
                    print('문이 열렸습니다.')
                break

            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


@app.route('/chemical_feed0')
def chemical_feed0():
    return Response(gen_chemical0(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_chswitch')
def check_chswitch():
    global ch_switch1, ch_switch0
    print('check_chswitch:clothes', ch_switch1)
    print('check_chswitch:Shoes', ch_switch0)
    return jsonify({'ch_switch1': ch_switch1, 'ch_switch0': ch_switch0})



#기본 실험실 웹캠
def gen_basicCoat():
    global b_switchCoat, b_switchShoes
    global bmodel
    bmodelcoat=None

    bmodelcoat = bmodel

    # bmodelCoat.names[0] ='labcoat'

    # 각 객체별 신뢰도 임계값 설정
    confidence_thresholds = {
        'labcoat': 0.8
    }

    # 객체별 바운딩 박스 색상 설정
    # 색상이 다르게나와서 원래랑 같게 나오도록 수정
    colors = {
        'labcoat': (255, 0, 0)  # Blue
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(0)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    # with app.app_context():  # 애플리케이션 컨텍스트 설정
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = bmodelcoat(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")

            # 모든 지정된 객체들이 3초 이상 연속으로 탐지되었는지 확인
            if all(continuous_detection[obj_name] for obj_name in confidence_thresholds):
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                b_switchCoat = True
                cv2.waitKey(3000)  # 3초간 대기
                if b_switchCoat and b_switchShoes == True:
                    print("문이 열립니다.")
                break

            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


@app.route('/basic_feed_coat')
def basic_feed_coat():
    return  Response(gen_basicCoat(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 기본 실험실 신발
def gen_basicShoes():
    global b_switchShoes, b_switchCoat
    global bmodel
    bmodelShoes = bmodel

    # bmodelShoes.names[1] = 'shoes'
    # bmodelShoes.names[2] = 'No-shoes'

    # 각 객체별 신뢰도 임계값 설정
    # confidence_thresholds = {
    #     'shoes': 0.65,
    #     'No-shoes': 0.9
    # }
    #
    # # 객체별 바운딩 박스 색상 설정
    # # 색상이 다르게나와서 원래랑 같게 나오도록 수정
    # colors = {
    #     'shoes': (0, 255, 255),  # Yellow
    #     'No-shoes': (180, 105, 255)  # Pink
    # }

    confidence_thresholds = {
        'shoes': 0.7,
        'No-shoes': 0.8
    }

    # 객체별 바운딩 박스 색상 설정
    colors = {
        'shoes': (255, 255, 0),  # Yellow
        'No-shoes': (255, 105, 180)  # Pink
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(1)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = bmodelShoes(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")

            # 모든 지정된 객체들(No-shoes 제외)이 3초 이상 연속으로 탐지되었는지 확인
            if continuous_detection['shoes']:
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                b_switchShoes = True
                cv2.waitKey(3000)  # 3초간 대기
                if b_switchCoat and b_switchShoes == True:
                    print("문이 열립니다.")
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


@app.route('/basic_feed_shoes')
def basic_feed_shoes():
    return Response(gen_basicShoes(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_bswitch')
def check_bswitch():
    global b_switchShoes, b_switchCoat
    print('check_bswitch:Coat', b_switchCoat)
    print('check_bswitch:Shoes', b_switchShoes)
    return jsonify({'b_switchCoat': b_switchCoat, 'b_switchShoes': b_switchShoes})


@app.route('/hardware')
def hardware():
    # global entry_name,  enable_switch
    global b_switchCoat, b_switchShoes
    face_switch = session.get('enable_switch', None)
    face_name = session.get('entry_name',None)
    print('enable_switch:', enable_switch)
    print('b_switchCoat:', b_switchCoat)
    print('b_switchShoes', b_switchShoes)
    print(face_name)
    if face_switch and ((b_switchCoat and b_switchShoes) or (ch_switch1 and ch_switch0)):
        # target_url = f"http://192.168.96.117:5000?entry_name={entry_name}"

         # target_url = f"http://http://192.168.82.117:5000?entry_name={face_name}"

        target_url = f"http://192.168.249.117:5000?entry_name={face_name}"
        return redirect(target_url)

        # return redirect("http://192.168.170.117:5000")
        # return render_template('hardware.html')
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('index'))
    # return redirect("http://192.168.96.117:5000")
    # return render_template('hardware.html')


if __name__ == '__main__':
    app.run(debug=True)
