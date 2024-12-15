from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input  # Import preprocess_input
import numpy as np

# Tải mô hình đã huấn luyện
model = load_model('rice_disease_predict.h5')  # Đảm bảo tệp này ở thư mục gốc dự án

# Danh sách nhãn
class_labels = ['Bacterial leaf blight', 'Brown spot', 'Healthy', 'Leaf blast', 'Leaf scald', 'Narrow brown spot']


def home(request):
    return render(request, 'home.html')


def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Lưu ảnh người dùng gửi lên
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)  # Sử dụng MEDIA_ROOT để lưu ảnh
        filename = fs.save(uploaded_file.name, uploaded_file)  # Lưu file và lấy tên file
        file_path = fs.path(filename)  # Đường dẫn đầy đủ đến file

        try:
            # Tiền xử lý ảnh
            target_size = (224, 224)
            img = load_img(file_path, target_size=target_size)  # Load ảnh từ đường dẫn đầy đủ
            img_array = img_to_array(img)
            img_tensor = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
            img_tensor = preprocess_input(img_tensor)  # Sử dụng preprocess_input để nhất quán với Colab

            # Dự đoán
            predictions = model.predict(img_tensor)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]
            confidence = predictions[0][predicted_class_index]

            # Xóa ảnh tạm
            fs.delete(filename)  # Xóa file sau khi xử lý

            # Trả kết quả
            context = {
                'predicted_class': predicted_class,
                'confidence': f"{confidence:.2f}"
            }
            return render(request, 'predict.html', context)
        except Exception as e:
            # Ghi log lỗi nếu có
            print(f"Lỗi trong quá trình dự đoán: {str(e)}")
            fs.delete(filename)
            return render(request, 'predict.html', {'error': 'Có lỗi xảy ra trong quá trình dự đoán. Vui lòng thử lại.'})

    # Khi request không phải POST hoặc không có file, trả về trang dự đoán trống
    return render(request, 'predict.html')
