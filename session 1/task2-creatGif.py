import imageio
from PIL import Image, ImageDraw
import cv2

def create_frame(width, height, frame_number):
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Vẽ đầu mèo
    draw.ellipse((50, 50, 250, 250), fill='gray')

    # Vẽ tai trái
    draw.ellipse((30, 120, 80, 170), fill='black')

    # Vẽ tai phải
    draw.ellipse((220, 120, 270, 170), fill='black')

    # Vẽ mắt trái
    draw.ellipse((85, 100, 115, 130), fill='black')

    # Vẽ mắt phải
    draw.ellipse((185, 100, 215, 130), fill='black')

    # Vẽ mũi
    draw.ellipse((145, 80, 155, 90), fill='black')

    # Vẽ miệng
    angle = 180 + frame_number % 180  # Mở miệng theo chu kỳ
    draw.arc((135, 110, 165, 130), start=0, end=angle, fill='black', width=2)

    return img

def create_gif(width, height, num_frames, output_path):
    frames = []
    for frame_number in range(num_frames):
        frame = create_frame(width, height, frame_number)
        frames.append(frame)

    # Lưu các hình ảnh đã tạo thành GIF
    imageio.mimsave(output_path, frames, duration=0.1)  # Đặt duration là thời gian hiển thị giữa các frame

gif_width = 300
gif_height = 200
num_frames = 60
output_path = "example.gif"

create_gif(gif_width, gif_height, num_frames, output_path)
cv2.waitKey()