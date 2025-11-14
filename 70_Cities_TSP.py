import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import time

class GeneticAlgorithmTSP:
    """
    使用模因算法(GA + 2-opt)解决TSP问题的类。
    """
    def __init__(self, distance_matrix, city_names, city_coords,
                 population_size, crossover_rate, mutation_rate,
                 tournament_size, elitism_size, num_generations):
        # 参数
        self.distance_matrix = distance_matrix
        self.city_names = city_names
        self.city_coords = city_coords
        self.num_cities = len(city_names)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.num_generations = num_generations
        
        # 内部状态
        self.population = []
        self.history = {'best_distance': [], 'avg_distance': [], 'best_paths': []}

    def _create_individual(self):
        cities = list(range(self.num_cities))
        random.shuffle(cities)
        return cities

    def _create_initial_population(self):
        self.population = [self._create_individual() for _ in range(self.population_size)]

    def _calculate_distance(self, individual):
        total_distance = 0
        for i in range(self.num_cities):
            start_city, end_city = individual[i], individual[(i + 1) % self.num_cities]
            total_distance += self.distance_matrix[start_city, end_city]
        return total_distance

    def _calculate_fitness(self, individual):
        return 1 / self._calculate_distance(individual)

    def _tournament_selection(self, pop_with_fitness):
        tournament_entrants = random.sample(pop_with_fitness, self.tournament_size)
        return max(tournament_entrants, key=lambda item: item[1])[0]

    def _order1_crossover(self, parent1, parent2):
        child = [None] * self.num_cities
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child[start:end] = parent1[start:end]
        parent2_genes = [gene for gene in parent2 if gene not in child]
        current_pos = end
        while None in child:
            if child[current_pos] is None:
                child[current_pos] = parent2_genes.pop(0)
            current_pos = (current_pos + 1) % self.num_cities
        return child

    def _swap_mutation(self, individual):
        idx1, idx2 = random.sample(range(self.num_cities), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def _inversion_mutation(self, individual):
        start, end = sorted(random.sample(range(self.num_cities), 2))
        sub_sequence = individual[start:end]
        individual[start:end] = sub_sequence[::-1]
        return individual

    def _two_opt_randomized(self, individual, max_iterations=150):
        best_tour = individual[:]
        for _ in range(max_iterations):
            i, j = sorted(random.sample(range(self.num_cities), 2))
            if j - i < 2: continue
            c1, c2 = best_tour[i], best_tour[i+1]
            c3, c4 = best_tour[j], best_tour[(j+1) % self.num_cities]
            if (self.distance_matrix[c1, c3] + self.distance_matrix[c2, c4]) < \
               (self.distance_matrix[c1, c2] + self.distance_matrix[c3, c4]):
                best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
        return best_tour

    def run(self):
        """运行一次完整的模因算法。"""
        self._create_initial_population()
        global_best_individual = self._create_individual()
        global_best_distance = self._calculate_distance(global_best_individual)

        for generation in range(self.num_generations):
            pop_with_fitness = [(ind, self._calculate_fitness(ind)) for ind in self.population]
            pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
            
            current_best_distance = self._calculate_distance(pop_with_fitness[0][0])
            if current_best_distance < global_best_distance:
                global_best_distance = current_best_distance
                global_best_individual = pop_with_fitness[0][0][:]
            
            avg_distance = sum(self._calculate_distance(ind) for ind in self.population)
            self.history['best_distance'].append(global_best_distance)
            self.history['avg_distance'].append(avg_distance / self.population_size)
            self.history['best_paths'].append(global_best_individual[:])

            next_generation = [pop_with_fitness[i][0][:] for i in range(self.elitism_size)]
            
            while len(next_generation) < self.population_size:
                parent1 = self._tournament_selection(pop_with_fitness)
                parent2 = self._tournament_selection(pop_with_fitness)
                child = self._order1_crossover(parent1, parent2)
                # 调整变异算法
                if random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)
                    # child = self._inversion_mutation(child)
                # 应用2-opt局部搜索
                child = self._two_opt_randomized(child)
                next_generation.append(child)
            
            self.population = next_generation
        
        return global_best_individual, global_best_distance

    def plot_convergence(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['best_distance'], label='Best Distance')
        plt.plot(self.history['avg_distance'], label='Average Distance')
        plt.title('Convergence Curve of the Best Run')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)

    def plot_best_path(self, best_path_indices, best_distance):
        plt.figure(figsize=(12, 8))
        coords = self.city_coords
        x_coords, y_coords = coords[:, 0], coords[:, 1]
        plt.scatter(x_coords, y_coords, c='red', zorder=2)
        for i, name in enumerate(self.city_names):
            plt.text(x_coords[i], y_coords[i], f' {name}', fontsize=9)
        ordered_coords = coords[best_path_indices]
        path_coords = np.vstack([ordered_coords, ordered_coords[0]])
        plt.plot(path_coords[:, 0], path_coords[:, 1], 'b-', zorder=1)
        start_node = ordered_coords[0]
        plt.scatter(start_node[0], start_node[1], c='lime', s=150, zorder=3, label='Start City')
        plt.title(f'Best Path Found (Distance: {best_distance:.4f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)

    def _update_animation(self, generation_num, line, title):
        path = self.history['best_paths'][generation_num]
        distance = self.history['best_distance'][generation_num]
        title.set_text(f'Generation: {generation_num + 1}/{self.num_generations}, Distance: {distance:.2f}')
        ordered_coords = self.city_coords[path]
        path_coords = np.vstack([ordered_coords, ordered_coords[0]])
        line.set_data(path_coords[:, 0], path_coords[:, 1])
        return line, title

    def animate_evolution(self, frame_skip=1, interval=50):
        fig, ax = plt.subplots(figsize=(12, 8))
        x_coords, y_coords = self.city_coords[:, 0], self.city_coords[:, 1]
        ax.scatter(x_coords, y_coords, c='red', zorder=2)
        line, = ax.plot([], [], 'b-', zorder=1)
        title = ax.set_title('Initializing animation...')
        ax.grid(True)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        frames_to_play = range(0, self.num_generations, frame_skip)
        ani = animation.FuncAnimation(fig, self._update_animation, frames=frames_to_play,
                                      fargs=(line, title), blit=True, interval=interval, repeat=False)
        return ani

def load_cities_from_excel_revised(filepath):
    try:
        df = pd.read_excel(filepath, header=None)
        coords, names = [], []
        counter = 1
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                if pd.notna(df.iloc[r, c]):
                    parts = str(df.iloc[r, c]).strip().split(',')
                    if len(parts) == 2:
                        try:
                            coords.append([float(parts[0]), float(parts[1])])
                            names.append(str(counter))
                            counter += 1
                        except ValueError: pass
        return (names, np.array(coords)) if coords else (None, None)
    except FileNotFoundError: return None, None
    except Exception: return None, None

def calculate_distance_matrix(coordinates):
    num_cities = len(coordinates)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            dist_matrix[i, j], dist_matrix[j, i] = dist, dist
    return dist_matrix

# --- 主执行程序 ---
if __name__ == '__main__':
    FILE_PATH = 'Locations_70_Citys.xlsx'
    city_names, city_coords = load_cities_from_excel_revised(FILE_PATH)

    if city_names is not None:
        dist_matrix = calculate_distance_matrix(city_coords)

        # --- 实验参数设置 ---
        NUM_RUNS = 30  # 独立运行次数
        
        # --- 模因算法参数 ---
        GA_PARAMS = {
            'population_size': 200,
            'crossover_rate': 0.85,
            'mutation_rate': 0.02,
            'tournament_size': 5,
            'elitism_size': 1,
            'num_generations': 1000
        }

        all_run_results = []
        start_time = time.time()
        
        print(f"开始执行 {NUM_RUNS} 次独立的模因算法运行...")
        print("="*60)

        for i in range(NUM_RUNS):
            print(f"--- 运行 {i + 1}/{NUM_RUNS} ---")
            
            ga_solver = GeneticAlgorithmTSP(
                distance_matrix=dist_matrix,
                city_names=city_names,
                city_coords=city_coords,
                **GA_PARAMS
            )
            
            final_path, final_distance = ga_solver.run()
            
            all_run_results.append({'distance': final_distance, 'path': final_path, 'solver_instance': ga_solver})
            print(f"运行 {i + 1} 完成。最短距离: {final_distance:.4f}\n")

        total_time = time.time() - start_time
        print(f"所有运行均已完成。总耗时: {total_time:.2f} 秒。")

        # --- 结果统计与分析 ---
        all_distances = [res['distance'] for res in all_run_results]
        best_run_index = np.argmin(all_distances)
        best_run = all_run_results[best_run_index]
        
        # 提取最佳运行的实例和结果
        best_solver = best_run['solver_instance']
        overall_best_path = best_run['path']
        overall_best_distance = best_run['distance']
        
        # 打印统计报告
        print("\n" + "="*60)
        print("           最终统计评估报告 ({}次运行)".format(NUM_RUNS))
        print("="*60)
        print(f"最佳结果 (Best):   {np.min(all_distances):.4f}")
        print(f"最差结果 (Worst):  {np.max(all_distances):.4f}")
        print(f"平均结果 (Mean):   {np.mean(all_distances):.4f}")
        print(f"标准差 (Std Dev):  {np.std(all_distances):.4f}")
        print("="*60)
        
        # --- 可视化最佳运行 ---
        print("\n正在生成最佳运行的可视化图表和动画...")

        # 1. 绘制收敛曲线
        best_solver.plot_convergence()

        # 2. 绘制最终路径
        best_solver.plot_best_path(overall_best_path, overall_best_distance)
        
        # 3. 生成并保存动画 (可选)
        animation_obj = best_solver.animate_evolution(frame_skip=5, interval=50)
        try:
            print("\n正在保存最佳运行的动画为 GIF...")
            animation_obj.save('best_run_evolution.gif', writer='pillow', fps=15, dpi=100)
            print("动画已成功保存为 'best_run_evolution.gif'")
        except Exception as e:
            print(f"保存为 GIF 失败: {e}")

        # 4. 显示所有图形窗口
        print("\n显示图形窗口...")
        plt.show()