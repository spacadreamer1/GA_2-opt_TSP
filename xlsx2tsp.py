import pandas as pd
import numpy as np

def load_cities_from_excel_revised(filepath):
    try:
        df = pd.read_excel(filepath, header=None)
        coords = []
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                if pd.notna(df.iloc[r, c]):
                    parts = str(df.iloc[r, c]).strip().split(',')
                    if len(parts) == 2:
                        try:
                            coords.append([float(parts[0]), float(parts[1])])
                        except ValueError: pass
        return np.array(coords) if coords else None
    except FileNotFoundError: return None
    except Exception: return None

def write_tsplib_file(coords, filename="cities.tsp", scale_factor=1000):
    """
    将坐标写入Concorde可读的TSPLIB文件。
    """
    num_cities = len(coords)
    with open(filename, 'w') as f:
        f.write(f"NAME: 70_cities_problem\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: 70 city problem for class project\n")
        f.write(f"DIMENSION: {num_cities}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords):
            # 格式: 节点ID X坐标 Y坐标
            scaled_x = int(x * scale_factor + 0.5)
            scaled_y = int(y * scale_factor + 0.5)
            f.write(f"{i + 1} {scaled_x} {scaled_y}\n")
        f.write("EOF\n")
    print(f"成功创建TSPLIB文件: '{filename}'")

if __name__ == '__main__':
    EXCEL_PATH = 'Locations_70_Citys.xlsx'
    city_coords = load_cities_from_excel_revised(EXCEL_PATH)

    if city_coords is not None:
        write_tsplib_file(city_coords, "my_70_cities.tsp", scale_factor=1000000)