# Pathfinding Arena: Player vs AI Simulator

Đây là một simulator trực quan được xây dựng bằng Pygame để so sánh các thuật toán tìm đường khác nhau (A*, Dijkstra, BFS, Greedy Best-First Search) và cho phép người dùng tự vẽ đường đi của mình để so tài với AI. Simulator hỗ trợ nhiều mẫu mê cung, các loại chướng ngại vật với chi phí di chuyển khác nhau, và hiệu ứng hình ảnh đơn giản.

## Tính năng

*   **So sánh thuật toán:** Trực quan hóa cách các thuật toán A*, Dijkstra, BFS, và Greedy BFS tìm đường trong cùng một môi trường.
*   **Hiển thị chi phí và số nút đã duyệt:** Cung cấp thông tin chi tiết về hiệu suất của mỗi thuật toán.
*   **Nhiều Agent:** Mỗi thuật toán được đại diện bởi một "xe" riêng với hình ảnh và màu sắc đường đi khác nhau.
*   **Ảnh nền và Sprites:** Sử dụng hình ảnh cho nền, tường, bẫy, cờ và xe để tăng tính trực quan.
*   **Hai loại chướng ngại vật:**
    *   **Tường (Wall):** Không thể đi qua.
    *   **Bẫy (Trap/Mud):** Có thể đi qua nhưng với chi phí cao hơn.
*   **Mê cung tùy chỉnh và có sẵn:**
    *   Người dùng có thể tự tạo mê cung bằng cách đặt tường, bẫy, điểm bắt đầu và kết thúc.
    *   Cung cấp một số mẫu mê cung được thiết kế sẵn để thử nghiệm nhanh.
*   **Chế độ vẽ đường đi của người chơi:**
    *   Người dùng có thể tự vẽ đường đi của mình từ điểm bắt đầu đến điểm kết thúc.
    *   Chi phí đường đi của người chơi được tính toán và so sánh với AI.
*   **Hiệu ứng hình ảnh:**
    *   Hiệu ứng nhấp nháy (thay đổi độ trong suốt - alpha) cho cờ Start/End và Bẫy.
    *   Hiệu ứng vệt bụi phía sau xe khi di chuyển.
*   **Xác định người chiến thắng:** Dựa trên tổng chi phí đường đi thấp nhất mà AI agent đạt được.

## Mô tả các Thuật toán Tìm Đường

Simulator này sử dụng các thuật toán tìm đường sau:

1.  **A* (A-Star Search):**
    *   **Mô tả:** Là một thuật toán tìm kiếm có thông tin (informed search), nổi tiếng với việc tìm đường đi ngắn nhất (hoặc chi phí thấp nhất) giữa hai điểm trên một đồ thị.
    *   **Cách hoạt động:** A\* kết hợp ưu điểm của Thuật toán Dijkstra (đảm bảo tìm đường đi tối ưu) và Greedy Best-First Search (sử dụng hàm heuristic để hướng dẫn tìm kiếm nhanh hơn). Nó đánh giá mỗi nút `n` bằng một hàm `f(n) = g(n) + h(n)`:
        *   `g(n)`: Chi phí thực tế từ nút bắt đầu đến nút `n`.
        *   `h(n)`: Chi phí ước tính (heuristic) từ nút `n` đến nút đích.
    *   **Đặc điểm:**
        *   **Tối ưu:** Đảm bảo tìm được đường đi có chi phí thấp nhất nếu hàm heuristic là "admissible" (không bao giờ đánh giá quá cao chi phí thực tế) và "consistent" (đơn điệu).
        *   **Hoàn chỉnh:** Sẽ tìm ra giải pháp nếu có.
        *   **Hiệu quả:** Thường hiệu quả hơn Dijkstra nhờ có heuristic định hướng.
    *   **Trong simulator:** Sử dụng heuristic Manhattan. Sẽ cố gắng tránh các ô "Bẫy" nếu có đường đi khác ít tốn kém hơn.

2.  **Dijkstra's Algorithm:**
    *   **Mô tả:** Một thuật toán kinh điển để tìm đường đi ngắn nhất từ một nút nguồn đến tất cả các nút khác trong một đồ thị có trọng số cạnh không âm.
    *   **Cách hoạt động:** Dijkstra mở rộng tìm kiếm từ nút bắt đầu, luôn chọn nút chưa được thăm có khoảng cách ngắn nhất từ nguồn để khám phá tiếp. Nó xây dựng một cây đường đi ngắn nhất.
    *   **Đặc điểm:**
        *   **Tối ưu:** Đảm bảo tìm được đường đi có chi phí thấp nhất.
        *   **Hoàn chỉnh:** Sẽ tìm ra giải pháp nếu có.
        *   **Không dùng heuristic:** Khám phá "mù quáng" ra mọi hướng, có thể duyệt qua nhiều nút hơn A\* trong các không gian lớn.
    *   **Trong simulator:** Được triển khai như A\* với hàm heuristic `h(n) = 0`. Sẽ tìm đường đi tối ưu về chi phí, tính cả chi phí của ô "Bẫy".

3.  **Breadth-First Search (BFS):**
    *   **Mô tả:** Một thuật toán tìm kiếm không có thông tin (uninformed search), khám phá đồ thị theo từng tầng (level by level).
    *   **Cách hoạt động:** Bắt đầu từ nút nguồn, BFS khám phá tất cả các nút lân cận trước, sau đó mới đến các nút lân cận của chúng, cứ thế tiếp tục. Nó sử dụng một hàng đợi (queue).
    *   **Đặc điểm:**
        *   **Tối ưu (về số bước):** Tìm đường đi có số lượng cạnh (bước) ít nhất. Nếu tất cả các cạnh có cùng trọng số (ví dụ: lưới chỉ có ô thường, không có bẫy), BFS cũng tìm được đường đi ngắn nhất về chi phí.
        *   **Hoàn chỉnh:** Sẽ tìm ra giải pháp nếu có.
        *   **Không quan tâm trọng số cạnh khi chọn đường:** Khi quyết định mở rộng nút nào, BFS không xét đến chi phí của các cạnh.
    *   **Trong simulator:** Sẽ tìm đường đi có ít ô nhất. Nó có thể đi qua ô "Bẫy" nếu đó là con đường có ít ô hơn, mặc dù chi phí thực tế của đường đi đó (được tính sau khi tìm thấy) có thể cao.

4.  **Greedy Best-First Search:**
    *   **Mô tả:** Một thuật toán tìm kiếm có thông tin, luôn cố gắng mở rộng nút mà nó tin là gần đích nhất dựa trên hàm heuristic.
    *   **Cách hoạt động:** Chỉ sử dụng hàm heuristic `h(n)` để đánh giá và chọn nút tiếp theo để mở rộng. Nó không quan tâm đến chi phí đã đi `g(n)`.
    *   **Đặc điểm:**
        *   **Nhanh (đôi khi):** Có thể tìm ra đường đi rất nhanh nếu heuristic tốt và không có nhiều "bẫy" cục bộ của heuristic.
        *   **Không tối ưu:** Không đảm bảo tìm được đường đi ngắn nhất hoặc chi phí thấp nhất.
        *   **Không hoàn chỉnh:** Có thể bị kẹt trong vòng lặp trên đồ thị có chu trình, hoặc đi vào ngõ cụt nếu heuristic không tốt.
    *   **Trong simulator:** Sử dụng heuristic Manhattan. Có thể bị "đánh lừa" bởi các đường đi trông có vẻ ngắn theo heuristic nhưng thực tế lại dài hoặc tốn nhiều chi phí (ví dụ: đi vào bẫy).

## Thư viện cần thiết

*   **Python 3.x** (Khuyến nghị 3.7 trở lên)
*   **Pygame:** Thư viện chính để tạo game và đồ họa.
    *   Cài đặt: `pip install pygame`

## Chuẩn bị trước khi chạy

1.  **Tải hoặc Clone Repository:** Lấy mã nguồn về máy của bạn.
2.  **Cài đặt Pygame:** Nếu chưa có, hãy chạy `pip install pygame` trong terminal hoặc command prompt.
3.  **Chuẩn bị hình ảnh (Sprites):**
    *   Tạo một thư mục có tên `images_game` trong cùng thư mục chứa file script chính (ví dụ: `main.py`).
    *   Đặt các file ảnh cần thiết vào thư mục `images_game`. Các file ảnh được chương trình mong đợi bao gồm:
        *   `ground.png` (Ảnh nền lớn cho toàn bộ lưới, ví dụ: 800x600 pixels)
        *   `wall.png` (Sprite cho ô tường, ví dụ: 32x32 pixels)
        *   `trap.png` (Sprite cho ô bẫy, ví dụ: 32x32 pixels)
        *   `start_flag.png` (Sprite cho điểm bắt đầu, ví dụ: 32x32 pixels)
        *   `end_flag.png` (Sprite cho điểm kết thúc, ví dụ: 32x32 pixels)
        *   `car_astar.png` (Sprite xe cho thuật toán A*, ví dụ: ~28x28 pixels, nên là PNG với nền trong suốt)
        *   `car_dijkstra.png` (PNG)
        *   `car_bfs.png` (PNG)
        *   `car_greedy.png` (PNG)
        *   `default_car.png` (Sprite xe mặc định nếu các ảnh xe cụ thể không tìm thấy, PNG)
    *   **Lưu ý:** Kích thước của các sprite tường, bẫy, cờ nên phù hợp với `CELL_SIZE` (mặc định là 32). Sprite xe nên nhỏ hơn một chút để vừa trong ô. Định dạng PNG với nền trong suốt được khuyến nghị cho các sprite (trừ ảnh nền). Nếu không có các file ảnh này, chương trình sẽ cố gắng sử dụng màu fallback.

## Cách sử dụng Simulator

1.  **Chạy chương trình:** Mở terminal hoặc command prompt, điều hướng đến thư mục chứa file script và chạy:
    ```bash
    python main.py
    ```
    (Hoặc tên file script của bạn nếu khác).

2.  **Giao diện chính:**
    *   Một lưới sẽ xuất hiện với ảnh nền (nếu có). Thông tin về chế độ hiện tại và các phím tắt sẽ được hiển thị ở phía trên và dưới màn hình.

3.  **Điều khiển và Phím tắt:**
    *   **Chuột trái:**
        *   **Đặt điểm Start:** Click vào một ô trống để đặt điểm bắt đầu (sprite cờ xanh). Chỉ có thể đặt một điểm Start.
        *   **Đặt điểm End:** Sau khi đặt Start, click vào một ô trống khác để đặt điểm kết thúc (sprite cờ xanh dương).
        *   **Đặt Tường/Bẫy:** Sau khi đặt Start và End, click vào các ô trống khác để đặt Tường (sprite tường) hoặc Bẫy (sprite bẫy), tùy thuộc vào chế độ xây dựng hiện tại.
        *   **Vẽ đường đi của người chơi (Player Path):** Khi ở chế độ "Player Draw ON", click chuột trái để thêm các điểm vào đường đi của bạn. Đường đi sẽ được tô màu cam nhạt.
    *   **Chuột phải:**
        *   Xóa một ô (Start, End, Tường, Bẫy) và đặt lại thành ô bình thường (hiển thị ảnh nền).
        *   Nếu ở chế độ "Player Draw ON", xóa điểm cuối cùng đã vẽ trong đường đi của người chơi.
    *   **Phím `W`:** Chuyển sang chế độ đặt **Tường (Wall)**.
    *   **Phím `T`:** Chuyển sang chế độ đặt **Bẫy (Trap)**.
    *   **Phím `P`:** Bật/Tắt chế độ **Vẽ đường đi của người chơi (Player Path)**.
        *   Khi bật, bạn có thể click chuột trái để vẽ đường đi từ điểm Start đã đặt.
        *   Nhấn `P` lần nữa hoặc `SPACE` để hoàn thành việc vẽ. Chi phí đường đi của bạn sẽ được tính và hiển thị.
    *   **Phím `SPACE`:**
        *   Nếu đang ở chế độ vẽ của người chơi, phím này sẽ hoàn thành việc vẽ và sau đó kích hoạt các thuật toán AI.
        *   Nếu không ở chế độ vẽ, phím này sẽ kích hoạt các AI agent tìm đường và di chuyển từ Start đến End hiện tại. Kết quả (chi phí, số nút duyệt) sẽ được in ra console và người chiến thắng (AI có chi phí thấp nhất) sẽ được hiển thị trên màn hình.
    *   **Phím `C`:** Xóa toàn bộ lưới, bao gồm Start, End, Tường, Bẫy, và tất cả các đường đi.
    *   **Phím số `1`, `2`, `3`, `4`:** Tải các mẫu mê cung được thiết kế sẵn. Tên mê cung hiện tại sẽ được hiển thị.

4.  **Quan sát:**
    *   Xem các "xe" (sprite đại diện cho mỗi thuật toán) di chuyển trên lưới, để lại vệt bụi.
    *   Theo dõi đường đi được tô màu của mỗi thuật toán.
    *   Quan sát hiệu ứng nhấp nháy của cờ Start/End và các ô Bẫy.
    *   Kiểm tra thông tin được in ra trên console để so sánh chi phí và hiệu suất (số nút đã duyệt).
    *   Xem thông báo người chiến thắng (AI) và chi phí đường đi của người chơi trên màn hình.

## Cấu trúc Code (Tổng quan)

*   **`main.py` (hoặc tên file của bạn):** Chứa vòng lặp chính của game, xử lý sự kiện, logic cập nhật và vẽ.
*   **Lớp `GridNode`:** Đại diện cho mỗi ô trên lưới, quản lý loại ô (thường, tường, bẫy, start, end), chi phí, sprite tương ứng, và hiệu ứng hình ảnh nhấp nháy.
*   **Lớp `Graph`:** Biểu diễn bản đồ lưới dưới dạng đồ thị cho các thuật toán tìm đường, với trọng số cạnh dựa trên chi phí của ô.
*   **Lớp `Agent`:** Đại diện cho mỗi "xe" AI, quản lý hình ảnh sprite, vị trí, góc quay, đường đi, chuyển động, và hiệu ứng vệt bụi.
*   **Hàm `load_sprites_and_background()`:** Tải tất cả các tài nguyên hình ảnh (nền, tường, bẫy, cờ, xe) vào đầu chương trình.
*   **Các hàm thuật toán tìm đường:** `a_star_search`, `dijkstra_search`, `bfs_search`, `greedy_bfs_search`.
*   **`MAZE_PATTERNS`:** Dictionary chứa định nghĩa của các mẫu mê cung (vị trí start, end, tường, bẫy).
*   **Hàm `load_maze()`:** Thiết lập lưới dựa trên một mẫu mê cung đã chọn.
*   **Hàm `calculate_path_cost()`:** Tính chi phí cho đường đi của người chơi.
*   **Các hàm vẽ:** `draw_main`, `draw_paths_and_agents`, `draw_grid_lines`.

## Mở rộng tiềm năng

*   Thêm nhiều thuật toán tìm đường khác (ví dụ: Bidirectional Search, IDA\*).
*   Cho phép người dùng điều chỉnh chi phí của ô bẫy hoặc tốc độ của agent.
*   Thêm các loại hiệu ứng hình ảnh hoặc âm thanh phức tạp hơn.
*   Lưu và tải các thiết kế mê cung của người dùng vào file.
*   Cải thiện AI của các agent (ví dụ: cho chúng tránh nhau nếu có nhiều agent trên cùng một đường đi).
*   Thêm menu và giao diện người dùng hoàn chỉnh hơn.

Chúc bạn có những trải nghiệm thú vị và học hỏi được nhiều điều với simulator này!
