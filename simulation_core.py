import numpy as np
import os
import matplotlib.pyplot as plt
import matlab.engine
import io
import re  # 确保导入 re 模块，虽然用于text_results的解析逻辑将改变，但可能其他地方仍用


class SignalSimulator:
    def __init__(self, output_base_dir='static/results', matlab_scripts_dir='.'):
        """
        初始化信号模拟器参数和输出目录。
        """
        self.grid_size = (68, 95)  # 栅格大小 [行数, 列数]
        self.num_stations = 5  # 基站数量
        self.num_iterations = 10000  # 蒙特卡洛迭代次数
        self.min_signal_strength = -100  # 最小信号强度要求(dBm)
        self.initial_power = 46  # 基站初始信号强度(dBm)

        # 频率和高度参数 (从 COST231_Walfisch_Ikegami_Model.m 迁移过来)
        self.freq = 2.6e9  # 信号频率(Hz) 2.6 GHz
        self.h_b = 15.25  # 基站天线有效高度(m)
        self.h_m = 1.5  # 移动台天线高度(m)
        self.h_av = 10  # 街道平均建筑物高度(m)
        self.w_av = 10  # 街道平均宽度(m)
        self.f_c = 0  # 街道方向修正因子 (0: 平行于街道方向, 1: 垂直于街道方向)

        self.output_base_dir = output_base_dir
        self.matlab_scripts_dir = matlab_scripts_dir

        # 确保输出目录存在
        os.makedirs(self.output_base_dir, exist_ok=True)

    def run_matlab_simulation(self, grid_csv_path, model_name='zuizhongban', params=None):
        """
        启动 MATLAB 引擎，运行指定模型脚本，并捕获结果。
        model_name: 'zuizhongban', 'COST231_Walfisch_Ikegami_Model', 'coverage_simulation'
        params: dict，若为 coverage_simulation 需包含 a, b, n, r, num_trials
        """
        import matlab.engine
        if params is None:
            params = {}
        eng = None
        results = {
            'success': False,
            'message': 'MATLAB 模拟过程中发生未知错误。',
            'image_paths': {},
            'text_output': '',  # 用于 simulation_stats.txt 的内容
            'matlab_log': ''  # 用于 MATLAB 控制台的原始输出
        }

        image_filenames = {
            'plot1': 'grid_and_stations.png',
            'plot2': 'signal_strength_map.png',
            'plot3': 'coverage_history.png',
            'plot4': 'signal_histogram.png'
        }
        stats_txt_filename = 'simulation_stats.txt'  # 统计信息文件名

        try:
            print("尝试启动 MATLAB 引擎...")
            eng = matlab.engine.start_matlab()
            print("MATLAB 引擎已启动。")

            if eng is None:
                results['message'] = "MATLAB 引擎启动失败。"
                print(results['message'])
                return results

            # 正确调用 eng 的方法，无需担心 FutureResult
            eng.cd(self.matlab_scripts_dir, nargout=0)
            print(f"MATLAB 工作目录已设置为: {self.matlab_scripts_dir}")

            # 检查脚本文件是否存在
            script_file = model_name + '.m' if model_name != 'zuizhongban' else 'zuizhongban.m'
            if not os.path.exists(os.path.join(self.matlab_scripts_dir, script_file)):
                results['message'] = f"错误: 未找到 '{script_file}' MATLAB 脚本文件。请确保它位于项目根目录。"
                print(results['message'])
                return results

            # 清理旧的 MATLAB 输出图片和统计文件
            for filename in list(image_filenames.values()) + [stats_txt_filename]:
                full_path = os.path.join(self.output_base_dir, filename)
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        print(f"已删除旧文件: {full_path}")
                    except OSError as e:
                        print(f"警告: 无法删除旧文件 {full_path}: {e}")

            eng.addpath(self.output_base_dir, nargout=0)
            print(f"已将 '{self.output_base_dir}' 添加到 MATLAB 路径。")

            matlab_stdout = io.StringIO()
            matlab_stderr = io.StringIO()

            # 动态调用不同模型
            if model_name == 'zuizhongban':
                print("正在运行 MATLAB 脚本 (zuizhongban.m)...")
                eng.zuizhongban(nargout=0, stdout=matlab_stdout, stderr=matlab_stderr)
            elif model_name == 'COST231_Walfisch_Ikegami_Model':
                print("正在运行 MATLAB 脚本 (COST231_Walfisch_Ikegami_Model.m)...")
                eng.COST231_Walfisch_Ikegami_Model(nargout=0, stdout=matlab_stdout, stderr=matlab_stderr)
            elif model_name == 'coverage_simulation':
                print("正在运行 MATLAB 脚本 (coverage_simulation.m)...")
                # 参数准备
                a = float(params.get('a', 100))
                b = float(params.get('b', 100))
                n = int(params.get('n', 5))
                r = float(params.get('r', 20))
                num_trials = int(params.get('num_trials', 100))
                eng.coverage_simulation(a, b, n, r, num_trials, nargout=0, stdout=matlab_stdout, stderr=matlab_stderr)
            else:
                results['message'] = f"未知模型: {model_name}"
                print(results['message'])
                return results

            results['matlab_log'] = matlab_stdout.getvalue() + matlab_stderr.getvalue()
            print("MATLAB 脚本运行完成。")
            print("MATLAB 日志 (已捕获):\n", results['matlab_log'])

            # 检查图片文件是否成功生成
            generated_image_paths = {}
            all_images_exist = True
            for key, filename in image_filenames.items():
                full_path = os.path.join(self.output_base_dir, filename)
                if os.path.exists(full_path):
                    generated_image_paths[key] = full_path
                    print(f"MATLAB 生成图片: {full_path}")
                else:
                    all_images_exist = False
                    print(f"错误: MATLAB 未生成图片: {full_path}")

            if not all_images_exist:
                results['message'] = "MATLAB 模拟完成，但部分结果图片未生成。请检查 MATLAB 脚本是否成功保存了所有图片。"
                results['success'] = False
                return results

            results['image_paths'] = generated_image_paths

            # === 新增：读取 simulation_stats.txt 的内容作为 text_output ===
            stats_file_path = os.path.join(self.output_base_dir, stats_txt_filename)
            if os.path.exists(stats_file_path):
                try:
                    with open(stats_file_path, 'r', encoding='utf-8') as f:
                        results['text_output'] = f.read()
                    print(f"已成功从 {stats_file_path} 读取统计信息。")
                except Exception as file_e:
                    results['text_output'] = f"无法读取统计信息文件: {file_e}"
                    print(f"读取 {stats_file_path} 失败: {file_e}")
            else:
                results['text_output'] = "未找到统计信息文件 (simulation_stats.txt)。"
                print(f"错误: 未找到统计信息文件: {stats_file_path}")

            print(f"最终 text_output (将发送到前端):\n{results['text_output']}")

            results['success'] = True
            results['message'] = 'MATLAB 模拟成功完成。'

        except Exception as e:
            results['message'] = f"MATLAB 引擎或模拟运行时发生错误: {e}"
            results['matlab_log'] += f"\nPython 捕获到错误: {e}"  # 附加错误到日志
            print(f"MATLAB 模拟失败: {results['message']}")
            import traceback  # 导入 traceback 模块
            traceback.print_exc()  # 打印详细的异常信息，帮助调试
        finally:
            if eng:
                print("关闭 MATLAB 引擎...")
                eng.quit()
                print("MATLAB 引擎已关闭。")
        return results


if __name__ == '__main__':
    print("--- 正在测试 SignalSimulator 类 ---")

    output_dir = os.path.join(os.getcwd(), 'static', 'uploads')
    os.makedirs(output_dir, exist_ok=True)
    test_csv_path = os.path.join(output_dir, 'school_grid.csv')

    if not os.path.exists(test_csv_path):
        test_grid_data = np.random.randint(0, 5, size=(68, 95))
        np.savetxt(test_csv_path, test_grid_data, delimiter=',', fmt='%d')
        print(f"已创建虚拟栅格数据文件: {test_csv_path}")

    simulator = SignalSimulator()
    print("\n--- 尝试运行 MATLAB 模拟 ---")
    results = simulator.run_matlab_simulation(test_csv_path)

    print("\n--- 模拟结果 ---")
    print(f"成功: {results['success']}")
    print(f"消息: {results['message']}")
    print(f"图片路径: {results['image_paths']}")
    print(f"文本输出:\n{results['text_output']}")
    print(f"MATLAB 日志:\n{results['matlab_log']}")
