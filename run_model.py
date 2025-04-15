from PIL import Image
import numpy as np
import torch
import cv2
from network import CNNModel

# global drawing state
drawing = False
last_x, last_y = -1, -1
brush_size = 20
canvas = np.zeros((256, 256), dtype=np.uint8)

txt_color = 255
btns = {
    'clear': {'x1': 10, 'y1': 10, 'x2': 70, 'y2': 40}
}

def draw_buttons(img):
    c = btns['clear']
    cv2.rectangle(img, (c['x1'], c['y1']), (c['x2'], c['y2']), color=128, thickness=-1)
    cv2.putText(img, 'Clear', (c['x1']+5, c['y2']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (txt_color), 2)

def clear_canvas():
    global canvas
    canvas[:] = 0

def load_model(path='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

def process_drawing(image):
    img = np.array(image)
    img = Image.fromarray((img).astype(np.uint8)).resize((28, 28))
    img = np.array(img)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img



def predict():
    model, device = load_model()
    image_path = 'debug_capture.png'
    img_pil = Image.open(image_path).convert('L')
    img_tensor = process_drawing(img_pil).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    print(f"Predicted digit: {pred}")

def draw_callback(event, x, y, flags, param):
    
    global drawing, last_x, last_y

    if event == cv2.EVENT_LBUTTONDOWN:
        c = btns['clear']
        if c['x1'] <= x <= c['x2'] and c['y1'] <= y <= c['y2']:
            clear_canvas()
            return
        drawing = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (last_x, last_y), (x, y), color=255, thickness=brush_size)
        last_x, last_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            cv2.imwrite('debug_capture.png', canvas)
            predict()


def main():
    cv2.namedWindow('Draw')
    cv2.setMouseCallback('Draw', draw_callback)
    print("Draw a digit, click Clear to reset, press 'q' to exit.")
    while True:
        display = canvas.copy()
        draw_buttons(display)
        cv2.imshow('Draw', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()