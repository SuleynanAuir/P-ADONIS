"""快速测试平滑合并效果"""
import subprocess
import sys

# 只处理image5（最小的图）
result = subprocess.run(
    [sys.executable, 'run_demo.py'],
    capture_output=False,
    text=True,
    cwd=r'C:\Users\Aiur\PEAN'
)

print("\n测试完成！")
print("请查看 demo_results 文件夹中最新的输出")
print("特别关注 SR 图像是否还有明显的横向条纹")
