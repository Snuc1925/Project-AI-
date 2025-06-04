import tkinter as tk
from tkinter import *
import os
from tkinter import messagebox
from test_main_easy import start_display as start_display_easy
from MCTS import start_display as start_display_medium
from test_main_hard import start_display as start_display_hard

class DotAndBoxGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Dots and Boxes")
        self.root.geometry("500x500+1000+300")
        self.root.resizable(False, False)
        icon_path = os.path.join(os.path.dirname(__file__), 'images', 'dotsandboxes.ico')
        self.root.iconbitmap(icon_path)

        self.selected_mode = None
        self.selected_level = None
        self.selected_ai1 = None
        self.selected_ai2 = None
        
        self.create_main_menu()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_main_menu(self):
        self.clear_window()

        container = tk.Frame(self.root)
        container.pack(expand=True, fill='both')

        container.columnconfigure(0, weight=1)

        container.rowconfigure(0, weight=2)
        container.rowconfigure(1, weight=1)
        container.rowconfigure(2, weight=1)
        container.rowconfigure(3, weight=1)
        container.rowconfigure(4, weight=2)

        title = tk.Label(container, text="Dots and Boxes", fg="red", font=("Comfortaa", 30))
        title.grid(row=0, column=0)

        mode = tk.Label(container, text="Chọn chế độ:", font=("VNI-Dom", 18))
        mode.grid(row=1, column=0)

        btn_person_vs_ai = tk.Button(container, text="Person vs AI", font=("VNI-Dom", 14),
                                     command=lambda: self.set_mode("Person vs AI"),
                                     bd="4", bg="light blue", fg="purple", width="10", pady="5")
        btn_person_vs_ai.grid(row=2, column=0)

        btn_ai_vs_ai = tk.Button(container, text="AI vs AI", font=("VNI-Dom", 14),
                                 command=lambda: self.set_mode("AI vs AI"),
                                 bd="4", bg="light green", fg="darkred", width="10", pady="5")
        btn_ai_vs_ai.grid(row=3, column=0)

    def set_mode(self, mode):
        self.selected_mode = mode
        if mode == "Person vs AI":
            self.select_level()
        elif mode == "AI vs AI":
            self.select_ai1()
        else:
            self.select_board_size()

    def set_level_and_continue(self, level):
        self.selected_level = level
        self.select_board_size()

    def blink_level5(self):
        if self.level5_btn and self.level5_btn.winfo_exists():
            current_color = self.level5_btn.cget("fg")
            next_color = "gray" if current_color == "white" else "white"
            self.level5_btn.config(fg=next_color)
            self.root.after(500, self.blink_level5)

    def select_level(self):
        self.clear_window()
        container = tk.Frame(self.root)
        container.pack(expand=True, fill="both")
        container.columnconfigure(0, weight=1)
        for i in range(6):
            container.rowconfigure(i, weight=1)

        label = tk.Label(container, text="Chọn độ khó:", font=("VNI-Dom", 16))
        label.grid(row=0, column=0)

        levels = [
            ("Level 1", "green", "white"),
            ("Level 2", "yellow", "black"),
            ("Level 3", "#ff9999", "blue"),
            ("Level 4", "#cc0000", "white")
        ]

        for i, (level, bg_color, fg_color) in enumerate(levels):
            btn = tk.Button(
                container, text=level, font=("VNI-Dom", 14),
                bg=bg_color, fg=fg_color, bd=5,
                command=lambda lvl=level: self.set_level_and_continue(lvl)
            )
            btn.grid(row=i+1, column=0, pady=2)

        # Level 5 with blinking effect
        self.level5_btn = tk.Button(
            container, text="Level 5", font=("VNI-Dom", 14),
            bg="#800080", fg="white", bd=5,
            command=lambda: self.set_level_and_continue("Level 5")
        )
        self.level5_btn.grid(row=5, column=0, pady=2)
        self.blink_level5()

        btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=self.create_main_menu)
        btn_back.pack(side=tk.BOTTOM, pady=10)
        
    def select_ai1(self):
        self.clear_window()

        container = tk.Frame(self.root)
        container.pack(expand=True, fill="both")

        container.columnconfigure(0, weight=1)
        for i in range(7):
            container.rowconfigure(i, weight=1)

        label = tk.Label(container, text="Chọn AI1:", font=("VNI-Dom", 16))
        label.grid(row=0, column=0)

        ai_options = [
            ("Random", "#8CFF9E"),                # xanh nhạt
            ("Intuitive Algorithm","#FFEB99"),   # vàng nhạt
            ("Genetic Algorithm","#FFA07A"),     # cam nhạt
            ("Alpha-Beta", "#FF6666"),            # đỏ nhạt
            ("MCTS", "#9933FF")                  # tím đậm
        ]

        for i, (ai_name,color) in enumerate(ai_options):
            btn = tk.Button(container, text=ai_name, font=("VNI-Dom", 14), bd=5,
                            command=lambda ai=ai_name: self.set_ai1(ai), width=15, bg=color)
            btn.grid(row=i+1, column=0, pady=2)

        btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=self.create_main_menu)
        btn_back.pack(side=tk.BOTTOM, pady=10)

    def set_ai1(self, ai_name):
        self.selected_ai1 = ai_name
        self.select_ai2()

    def select_ai2(self):
        self.clear_window()

        container = tk.Frame(self.root)
        container.pack(expand=True, fill="both")

        container.columnconfigure(0, weight=1)
        for i in range(7):
            container.rowconfigure(i, weight=1)

        label = tk.Label(container, text="Chọn AI2:", font=("VNI-Dom", 16))
        label.grid(row=0, column=0)

        ai_options = [
            ("Random", "#8CFF9E"),                # xanh nhạt
            ("Intuitive Algorithm","#FFEB99"),   # vàng nhạt
            ("Genetic Algorithm","#FFA07A"),     # cam nhạt
            ("Alpha-Beta", "#FF6666"),            # đỏ nhạt
            ("MCTS", "#9933FF")                  # tím đậm
        ]
        for i, (ai_name, color) in enumerate(ai_options):
            btn = tk.Button(container, text=ai_name, font=("VNI-Dom", 14), bd=5,bg=color,
                            width=15, command=lambda ai=ai_name: self.set_ai2(ai))
            btn.grid(row=i+1, column=0, pady=2)

        btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=self.select_ai1)
        btn_back.pack(side=tk.BOTTOM, pady=10)

    def set_ai2(self, ai_name):
        self.selected_ai2 = ai_name
        self.select_board_size()

    def select_board_size(self):
        self.clear_window()
        
        container = tk.Frame(self.root)
        container.pack(expand=True, fill="both")

        container.columnconfigure(0, weight=1)

        container.rowconfigure(0, weight=2)  
        container.rowconfigure(1, weight=1) 
        container.rowconfigure(2, weight=1) 
        container.rowconfigure(3, weight=1)  
        container.rowconfigure(4, weight=1)
        container.rowconfigure(5, weight=1)
        container.rowconfigure(6, weight=2) 
        
        label = tk.Label(container, text= "Chọn kích thước bảng:", font=("VNI-Dom", 16))
        label.grid(column=0,row=1)
        
        sizes = ["3x3", "4x4", "4x5","5x5"]
        for size in sizes:
            if (size == "3x3"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s,self.selected_mode), bd="5", bg="Green", fg="Yellow") 
                btn.grid(column=0,row=2)
            if (size == "4x4"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s, self.selected_mode), bd="5", bg="Yellow")
                btn.grid(column=0,row=3)
            if (size == "4x5"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s, self.selected_mode), bd="5", bg="#F4A950", fg="blue")
                btn.grid(column=0,row=4)
            if (size == "5x5"): 
                btn = tk.Button(container, text=size, font=("VNI-Dom", 14),
                            command=lambda s=size: self.initiate_game_start(s, self.selected_mode), bd="5", bg="#ff5b61", fg="light yellow")
                btn.grid(column=0,row=5)
                
                def blink():
                    current_color = btn.cget("fg")
                    new_color = "cyan" if current_color == "light yellow" else "light yellow"
                    btn.config(fg=new_color)
                    btn.after(500, blink)

                blink()  # Hiệu ứng nhấp nháy
            # btn.pack(pady=20)

        if self.selected_mode == "AI vs AI":
            btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=lambda: self.select_ai2(self.selected_ai1))
        elif self.selected_mode == "Person vs AI":
            btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=self.select_level)
        else:
            btn_back = tk.Button(self.root, text="← Quay lại", font=("VNI-Dom", 10), command=self.create_main_menu)
        btn_back.pack(side=tk.BOTTOM, pady=10)

    def initiate_game_start(self, board_size_str, mode):
        try:
            self.root.withdraw()
            if board_size_str == "3x3":
                real_board_size = "4x4"
            elif board_size_str == "4x4":
                real_board_size = "5x5"
            elif board_size_str == "4x5":
                real_board_size = "5x6"
            elif board_size_str == "5x5":
                real_board_size = "6x6"
            else:
                real_board_size = board_size_str
            if mode == "Person vs AI":
                if self.selected_level == "Level 1":
                    start_display_easy(real_board_size, mode, self.on_back_to_menu)
                elif self.selected_level == "Level 2":
                    start_display_medium(real_board_size, mode, self.on_back_to_menu)
                elif self.selected_level == "Level 3":
                    start_display_hard(real_board_size, mode, self.on_back_to_menu)
                else:
                    raise ValueError("Chưa chọn mức độ khó.")
            else:
                start_display_easy(real_board_size, mode, self.on_back_to_menu)
        except Exception as e:
            messagebox.showerror("Error", f"Could not start game: {e}")
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.deiconify()
                    self.create_main_menu()
            except tk.TclError:
                print("Tkinter window was closed.")
    def on_back_to_menu(self):
        self.root.deiconify()
        self.create_main_menu()
    
# Chạy chương trình
if __name__ == "__main__":
    root = tk.Tk()
    app = DotAndBoxGame(root)
    root.mainloop()
