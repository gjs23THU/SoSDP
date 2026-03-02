import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _pick_xy_columns(frame: pd.DataFrame):
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if len(numeric_cols) < 2:
        raise ValueError("表格中可用于绘图的数值列少于2列")
    return numeric_cols[0], numeric_cols[1]


def visualize_excel(excel_path: str, output_dir: str, annotate: bool = False):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"文件不存在: {excel_path}")

    os.makedirs(output_dir, exist_ok=True)

    workbook = pd.ExcelFile(excel_path)
    if len(workbook.sheet_names) == 0:
        raise ValueError("Excel 文件没有可读取的 sheet")

    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=0)
        frame = frame.dropna(how="all").reset_index(drop=True)

        if frame.empty:
            continue

        x_col, y_col = _pick_xy_columns(frame)

        plt.figure(figsize=(6, 6))
        plt.scatter(frame[x_col], frame[y_col], s=35)

        if annotate:
            for idx, (x_val, y_val) in enumerate(zip(frame[x_col], frame[y_col])):
                plt.text(x_val, y_val, str(idx), fontsize=8)

        plt.title(f"{sheet_name}: {x_col} vs {y_col}")
        plt.xlabel(str(x_col))
        plt.ylabel(str(y_col))
        plt.axis("equal")
        plt.grid(alpha=0.25)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{sheet_name}.png")
        plt.savefig(output_path, dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="读取 A.xlsx 并生成每个 sheet 的散点图")
    parser.add_argument("--input", default="/home/adminstrator/文档/gjs23/SoSDP/SoSDP/A.xlsx", help="输入 Excel 文件路径")
    parser.add_argument("--output", default="excel_vis", help="图片输出目录")
    parser.add_argument("--annotate", action="store_true", help="是否给点加编号")
    args = parser.parse_args()

    visualize_excel(args.input, args.output, annotate=args.annotate)
    print(f"可视化完成，输出目录: {args.output}")


if __name__ == "__main__":
    main()
