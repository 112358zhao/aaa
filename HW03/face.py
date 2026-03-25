import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import face_recognition

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="人脸检测与识别系统",
    page_icon="😀",
    layout="wide"
)

# ---------- 初始化会话状态 ----------
if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = []
if "known_names" not in st.session_state:
    st.session_state.known_names = []
if "face_lib_loaded" not in st.session_state:
    st.session_state.face_lib_loaded = False


# ---------- 人脸库管理 ----------
def load_face_library(lib_path="face_library.pkl"):
    """加载已知人脸库文件"""
    if os.path.exists(lib_path):
        try:
            with open(lib_path, "rb") as f:
                data = pickle.load(f)
                st.session_state.known_encodings = data.get("encodings", [])
                st.session_state.known_names = data.get("names", [])
                st.session_state.face_lib_loaded = True
            return True
        except Exception as e:
            st.error(f"加载人脸库失败: {e}")
            return False
    return False


def save_face_library(lib_path="face_library.pkl"):
    """保存已知人脸库到文件"""
    data = {
        "encodings": st.session_state.known_encodings,
        "names": st.session_state.known_names
    }
    try:
        with open(lib_path, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"保存人脸库失败: {e}")
        return False


def add_face_to_library(name, encoding):
    """添加新的人脸到库中"""
    # 简单去重：避免添加同名同编码（可选）
    st.session_state.known_names.append(name)
    st.session_state.known_encodings.append(encoding)
    save_face_library()
    st.success(f"已添加 {name} 到人脸库")


def clear_face_library():
    """清空人脸库"""
    st.session_state.known_encodings = []
    st.session_state.known_names = []
    save_face_library()
    st.session_state.face_lib_loaded = True  # 已清空但仍是加载状态
    st.success("人脸库已清空")


# ---------- 人脸检测与识别函数 ----------
def detect_and_recognize(image):
    """
    检测图像中的人脸，并进行识别（如果人脸库非空）
    返回: (带标注的图像, 人脸数量, 识别结果列表)
    """
    # 转换 PIL 图像为 RGB numpy 数组
    img_array = np.array(image.convert("RGB"))
    
    # 检测人脸位置
    face_locations = face_recognition.face_locations(img_array)
    
    if not face_locations:
        return image, 0, []
    
    # 获取人脸编码
    face_encodings = face_recognition.face_encodings(img_array, face_locations)
    
    # 识别结果
    recognition_results = []
    
    # 如果有已知人脸库，进行识别
    if st.session_state.known_encodings and len(st.session_state.known_encodings) > 0:
        for i, encoding in enumerate(face_encodings):
            # 计算与所有已知人脸的欧氏距离
            distances = face_recognition.face_distance(st.session_state.known_encodings, encoding)
            best_match_idx = np.argmin(distances)
            best_distance = distances[best_match_idx]
            
            # 设置阈值，通常 0.6 以下认为是同一个人（可根据需要调整）
            threshold = 0.6
            if best_distance < threshold:
                name = st.session_state.known_names[best_match_idx]
                confidence = 1 - best_distance  # 置信度转换
            else:
                name = "未知"
                confidence = 0.0
            recognition_results.append((name, confidence))
    else:
        # 无库时，所有检测到的人脸标记为"未知"
        recognition_results = [("未知", 0.0) for _ in face_encodings]
    
    # 绘制标注
    annotated_image = draw_face_boxes(image, face_locations, recognition_results)
    
    return annotated_image, len(face_locations), recognition_results


def draw_face_boxes(image, face_locations, recognition_results):
    """
    在图像上绘制人脸框和标签
    """
    # 转换为可绘制的 RGB 模式
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    draw = ImageDraw.Draw(image)
    
    # 尝试加载中文字体，如果没有则使用默认字体
    try:
        # 对于 Windows，可使用 simhei.ttf；Linux/Mac 可能需要安装中文字体
        font = ImageFont.truetype("simhei.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    for (top, right, bottom, left), (name, conf) in zip(face_locations, recognition_results):
        # 绘制矩形框
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
        
        # 绘制标签背景
        if name == "未知":
            label = "未知"
            color = "gray"
        else:
            label = f"{name} ({conf:.2f})"
            color = "green"
        
        # 获取文字尺寸（粗略方法）
        bbox = draw.textbbox((left, top), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制背景矩形
        draw.rectangle(
            [(left, top - text_height - 5), (left + text_width + 5, top)],
            fill=color
        )
        # 绘制文字
        draw.text((left + 2, top - text_height - 3), label, fill="white", font=font)
    
    return image


# ---------- Streamlit UI ----------
def main():
    st.title("😀 人脸检测与识别系统")
    st.markdown("基于 `face_recognition` 和 `Streamlit` 实现的人脸检测与识别 Web 应用")
    
    # 侧边栏：人脸库管理
    with st.sidebar:
        st.header("📁 人脸库管理")
        
        # 加载默认人脸库（如果有）
        if not st.session_state.face_lib_loaded:
            if load_face_library():
                st.info(f"已加载人脸库，当前共 {len(st.session_state.known_names)} 人")
            else:
                st.info("未找到现有人脸库，可添加新样本")
        
        # 显示当前人脸库
        if st.session_state.known_names:
            st.subheader("已知人脸列表")
            for i, name in enumerate(st.session_state.known_names):
                st.write(f"{i+1}. {name}")
        else:
            st.info("当前人脸库为空，请添加人脸样本")
        
        st.markdown("---")
        st.subheader("➕ 添加人脸样本")
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("姓名", key="new_name")
        with col2:
            # 上传人脸图片用于注册
            reg_image = st.file_uploader("上传人脸照片", type=["jpg", "jpeg", "png"], key="reg")
        
        if st.button("添加至人脸库", use_container_width=True):
            if not new_name.strip():
                st.error("请输入姓名")
            elif reg_image is None:
                st.error("请上传照片")
            else:
                # 读取图片并检测人脸
                img = Image.open(reg_image)
                img_array = np.array(img.convert("RGB"))
                face_locations = face_recognition.face_locations(img_array)
                if len(face_locations) == 0:
                    st.error("未检测到人脸，请上传包含清晰人脸的照片")
                elif len(face_locations) > 1:
                    st.warning("检测到多张人脸，将使用第一张人脸进行注册")
                    # 取第一张人脸
                    encoding = face_recognition.face_encodings(img_array, face_locations)[0]
                    add_face_to_library(new_name.strip(), encoding)
                else:
                    encoding = face_recognition.face_encodings(img_array, face_locations)[0]
                    add_face_to_library(new_name.strip(), encoding)
        
        if st.button("清空人脸库", use_container_width=True):
            clear_face_library()
    
    # 主区域：上传图片并检测
    st.header("📷 图片上传与检测")
    
    # 示例图片选项
    example_images = {
        "无": None,
        "示例1 (多人)": "examples/multi_face.jpg",
        "示例2 (单人)": "examples/single_face.jpg"
    }
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_example = st.selectbox("选择示例图片", list(example_images.keys()))
        uploaded_file = st.file_uploader("或者上传图片", type=["jpg", "jpeg", "png"])
    
    # 确定使用的图片
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif selected_example != "无" and example_images[selected_example]:
        if os.path.exists(example_images[selected_example]):
            image = Image.open(example_images[selected_example])
        else:
            st.warning(f"示例图片 {example_images[selected_example]} 不存在，请创建 examples 目录并放置示例图片")
            image = None
    else:
        # 显示占位提示
        st.info("请选择示例图片或上传图片")
    
    if image is not None:
        # 显示原图
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原图")
            st.image(image, use_container_width=True)
        
        # 执行检测与识别
        with st.spinner("正在检测人脸..."):
            annotated_img, face_count, results = detect_and_recognize(image)
        
        with col2:
            st.subheader("检测结果")
            st.image(annotated_img, use_container_width=True)
        
        # 显示检测信息
        st.success(f"检测到 {face_count} 张人脸")
        if face_count > 0:
            st.write("识别详情：")
            for i, (name, conf) in enumerate(results):
                if name == "未知":
                    st.write(f"人脸 {i+1}: 未知")
                else:
                    st.write(f"人脸 {i+1}: {name} (置信度: {conf:.2f})")
        else:
            st.warning("未检测到人脸")


if __name__ == "__main__":
    main()