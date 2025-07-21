from flask import Flask, render_template, request, jsonify, url_for
import os
import uuid
import re

# 导入图像转栅格的Python函数
from image_to_grid_python import image_to_grid_python
# 导入信号模拟器类
from simulation_core import SignalSimulator

app = Flask(__name__)

# 配置上传文件保存的目录
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 确保目录存在
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 配置模拟结果图片保存的目录
RESULTS_FOLDER = os.path.join(app.root_path, 'static', 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)  # 确保目录存在
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# 定义允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 定义Windows文件系统不允许的字符
INVALID_WINDOWS_CHARS = r'[<>:"/\\|?*\x00-\x1F]'  # 包括控制字符


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """
    渲染主页。
    """
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    """
    处理图像上传和信号模拟请求。
    """
    if 'gridImage' not in request.files:
        return jsonify({'success': False, 'message': '没有文件部分'}), 400

    file = request.files['gridImage']

    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400

    if file and allowed_file(file.filename):
        # 1. 保存上传的图片
        base_filename, file_extension = os.path.splitext(file.filename)

        # 清理文件名中的非法字符
        cleaned_base_filename = re.sub(INVALID_WINDOWS_CHARS, '', base_filename)
        if not cleaned_base_filename:
            cleaned_base_filename = "uploaded_image"

        filename = str(uuid.uuid4()) + file_extension.lower()

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        print(f"尝试保存文件到路径: {image_path}")
        print(f"image_path (repr): {repr(image_path)}")

        try:
            file.save(image_path)
            print(f"文件已成功保存到: {image_path}")
        except OSError as e:
            print(f"保存文件时发生 OSError: {e}")
            return jsonify(
                {'success': False, 'message': f'保存文件失败: {e}. 请检查文件名是否包含特殊字符或路径是否过长。'}), 500

        # 2. 调用 image_to_grid_python 将图片转换为栅格数据
        target_rows = 68
        target_cols = 95
        output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'school_grid.csv')

        grid_data = image_to_grid_python(image_path, target_rows, target_cols, output_dir=app.config['UPLOAD_FOLDER'])

        if grid_data is None:
            print("图像转换为栅格数据失败。")
            return jsonify({'success': False, 'message': '图像转换失败，请检查图片格式和内容。'}), 500

        print("图像已成功转换为栅格数据。")

        # 3. 调用 SignalSimulator 进行 MATLAB 信号模拟
        simulator = SignalSimulator()

        # 新增：获取模型名和参数
        model_name = request.form.get('model', 'zuizhongban')
        params = {}
        if model_name == 'coverage_simulation':
            params['a'] = request.form.get('paramA', 100)
            params['b'] = request.form.get('paramB', 100)
            params['n'] = request.form.get('paramN', 5)
            params['r'] = request.form.get('paramR', 20)
            params['num_trials'] = request.form.get('paramTrials', 100)

        matlab_results = simulator.run_matlab_simulation(output_csv_path, model_name=model_name, params=params)

        if not matlab_results['success']:
            print(f"MATLAB 模拟失败: {matlab_results['message']}")
            return jsonify(matlab_results), 500

        # 将文件系统路径转换为可供前端访问的URL
        image_urls_for_frontend = {}
        for key, path_rel in matlab_results['image_paths'].items():
            # 移除 static/ 前缀并确保路径使用正斜杠
            static_path = os.path.relpath(path_rel, app.static_folder)
            # === 关键修改：将所有反斜杠替换为正斜杠，以适应URL规范 ===
            url_friendly_path = static_path.replace('\\', '/')
            image_urls_for_frontend[key] = url_for('static', filename=url_friendly_path)

        return jsonify({
            'success': True,
            'message': '文件上传成功，图像已处理，信号模拟完成。',
            'image_urls': image_urls_for_frontend,
            'text_results': matlab_results['text_output'],
            'matlab_log': matlab_results['matlab_log']
        })

    return jsonify({'success': False, 'message': '文件类型不允许'}), 400


if __name__ == '__main__':
    app.run(debug=True)