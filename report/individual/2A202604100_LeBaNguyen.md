# Báo cáo Day 7 — Agentic RAG

Lê Bá Nguyên · MSSV 2A202604100 · Lớp L3B

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang, số chunk:** *The World Bank Group Annual Report 2026* (FY26, tiếng Anh, CC BY-NC-ND 3.0 IGO), 42 trang nội dung (p3–44). Dùng lại index Day 6 (repo `K5-L3B-Day06-LeBaNguyen-2A202604100-DataIngestionEngineering`). Strategy B (chia theo đoạn, tối đa 1000 ký tự): **255 chunk**; strategy A (500 ký tự, overlap 50): 433 chunk. Agent và mọi thí nghiệm dưới đây dùng strategy B.
- **Vector store, embedding model, số chiều:** PostgreSQL 16 + pgvector 0.8.7 (Docker, tìm cosine chính xác, không có chỉ mục ANN); `intfloat/multilingual-e5-small`, 384 chiều, chạy local bằng sentence-transformers (có tiền tố `query:`/`passage:`), chi phí embedding 0 USD. Day 7 kết nối qua `DATABASE_URL` trong `.env` (không nộp). Container Day 6 được chạy trên cổng 55600 vì cổng 55433 nằm trong dải Windows giữ chỗ.
- **Agent model, cách chạy, chia dev/test:** Gemini (`gemini-3.5-flash-lite`, temperature 0, SDK `google-genai`) điều khiển vòng lặp `search → context → answer/search tiếp` (`starter/agentic.py`, `controller.py`). Benchmark 26 câu: **dev 18** (16 có đáp án + 2 thiếu bằng chứng), **test 8** (6 có đáp án + 2 thiếu bằng chứng). Chia dev/test trước khi thử nghiệm; 6 câu có trường `filter` (4 dev, 2 test) nhưng **filter không được dùng** trong các thí nghiệm này (xem giới hạn).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold_source_id`, `gold_page`, `evidence_quote`; một chunk là *hit* khi đúng nguồn, đúng trang và chứa nguyên văn evidence. Câu thiếu bằng chứng (u01–u04) được kiểm tra bằng cách quét cả 255 chunk theo từ khóa (lương Chủ tịch, hãng hàng không, tiền mã hóa Kenya, xe điện): không có nội dung nào liên quan (từ "compensation" chỉ xuất hiện trong ngữ cảnh ICSID). Nhãn câu trả lời/citation/context do **Claude chấm sau khi đọc từng câu trả lời và chunk được trích dẫn**; người học cần rà soát lại (xem giới hạn).

## Retrieval và thử nghiệm

Tất cả dòng dưới đây dùng strategy B, truy vấn bằng **câu hỏi gốc** (không viết lại), không filter. "Context Evidence Rate" là tỷ lệ câu mà context sau `pack_context` còn chứa chunk bằng chứng. Hit@3 luôn tính trên top-3 dù top-k lớn hơn. `n/N` và thời gian trung vị mỗi truy vấn (ms).

**Dev (N = 16 câu có đáp án):**

| Cấu hình | Phương pháp, top-k, budget | Giả thuyết / thay đổi | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (ms) |
|---|---|---|---|---|---|---|
| Baseline | Dense, top-3, budget 800 | Mốc so sánh | 14/16 (87.5%) | 14/16 (87.5%) | **2/16 (12.5%)** | 81 |
| Phương pháp 2 | Lexical TF-IDF, top-3, budget 800 | Từ khóa có thể bắt được số liệu/bảng | 11/16 (68.8%) | 12/16 (75.0%) | 2/16 | 2 |
| Phương pháp 3 | Hybrid RRF (dense + lexical), top-3, budget 800 | Kết hợp hai nguồn xếp hạng | 13/16 (81.3%) | 14/16 (87.5%) | 2/16 | 90 |
| Cải thiện 1 | Dense, top-3, **budget 2000** | Chunk bằng chứng ~1000 ký tự không vừa budget 800 nên bị bỏ | 14/16 | 14/16 | **14/16 (87.5%)** | 79 |
| Cải thiện 2 | Dense, **top-5, budget 3500** | Lấy thêm ứng viên | 14/16 | 14/16 | 14/16 (87.5%) | 89 |
| Cấu hình đã chọn | **Hybrid RRF, top-10, budget 10000** | Mở rộng ứng viên để cứu chunk rank 5–10 (q08, q16) | 13/16 | 14/16 | 15/16 (93.8%) | 47 |

Phương pháp 2 và 3 ở budget 2000: lexical 11/16 · 12/16 · context 11/16; hybrid 13/16 · 14/16 · context 13/16. Dense top-10/budget 10000 và lexical top-10 cũng đạt context 15/16.

**Test (N = 6 câu có đáp án), chạy sau khi chốt cấu hình:**

| Cấu hình | Phương pháp, top-k, budget | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (ms) |
|---|---|---|---|---|---|
| Baseline | Dense, top-3, budget 800 | 5/6 (83.3%) | 6/6 (100%) | **0/6 (0%)** | 40 |
| Baseline | Lexical, top-3, budget 800 | 5/6 | 6/6 | 0/6 | 1 |
| Baseline | Hybrid, top-3, budget 800 | 5/6 | 6/6 | 0/6 | 41 |
| Dense, budget 2000 | Dense, top-3, budget 2000 | 5/6 | 6/6 | 5/6 (83.3%) | 58 |
| **Cấu hình đã chọn** | **Hybrid, top-10, budget 10000** | 5/6 | 6/6 | **6/6 (100%)** | 40 |

Cả ba phương pháp hòa nhau trên test (5/6 · 6/6), nên với N = 6 test không phân biệt được chất lượng xếp hạng; trên dev (N = 16) dense dẫn đầu Hit@1 (14 so với 13 hybrid và 11 lexical). Điểm khác biệt lớn nhất nằm ở **context**, không nằm ở retrieval.

**Chấm context 0/1/2 trên 6 câu test (file `starter/results/context-review-test.json`):** baseline (top-3, budget 800): sáu câu đều 0, trung bình 0.0. Dense budget 2000: năm câu 2 và một câu 1 (q19: context có con số 16,000 nhưng thiếu câu gốc "create, protect, and safeguard"), trung bình 1.83.

### Hai lỗi dev (cộng thêm một lỗi thứ ba), nguyên nhân, cách thử và lý do chọn cấu hình

**Lỗi 1 — Context rỗng hoặc thiếu evidence do budget 800.** Retrieval tìm đúng (Hit@3 14/16) nhưng chỉ 2/16 câu có evidence trong context. Nguyên nhân: `pack_context` bỏ cả chunk nếu không vừa budget, mà chunk bằng chứng dài ~1000 ký tự cộng phần tiêu đề nguồn. Trên test, context rỗng ở 5/6 câu. Cách thử: budget 2000 và top-5/budget 3500. Trước/sau: context evidence 2/16 → 14/16 (dev); test 0/6 → 5/6 (budget 2000). Giả thuyết đúng.

**Lỗi 2 — q08 và q16 bị từ chối nhầm.** Nguyên nhân (đo bằng cách xếp hạng chunk đúng trong top-20, script `starter/rank_dev.py`):

| Câu | Dense | Lexical | Hybrid | Ghi chú |
|---|---|---|---|---|
| q08 (Mission 300) | hạng 5 | 12 | 6 | Ngay ngoài top-3 |
| q16 (net income IFC) | **không có trong top-20** | 7 | 10 | Evidence là một dòng bảng số liệu ("Net Income (Loss) 2,038") nên embedding ngữ nghĩa khó khớp; chỉ lexical bắt được |

Ở cả hai câu, agent lặp lại gần như cùng một query 3 lần, kết quả giống hệt. Giả thuyết A: "cấm lặp query sẽ giúp" (E3). **Bị bác bỏ:** q16 vẫn từ chối; q08 trả lời sai (nói Mission 300 là một trong 10 ưu tiên chiến lược thay vì mục tiêu); q14 mất evidence ở context cuối. Giả thuyết B: "mở rộng ứng viên (top-k lớn) cứu được chunk đứng hạng 5–10, và hybrid cứu được bảng nhờ thành phần lexical". **Được xác nhận:** dense top-8 sửa q08, hybrid top-10 sửa cả q08 lẫn q16.

**Lỗi 3 (phát hiện khi chạy E1/E2) — Định dạng citation `[id1, id2]` bị bác.** Khi context lớn, model trích hai chunk và viết `[id1, id2]` trong một cặp ngoặc; `run_agent` kiểm tra từng `[id]` riêng nên bỏ câu trả lời *đúng* và trả về "không đủ bằng chứng" (q01, q02 bị thoái lui; đã xác nhận bằng cách in đầu ra thô của model). Cách sửa: thêm quy tắc vào prompt ("mỗi chunk-id trong cặp ngoặc riêng"), không nới lỏng bộ kiểm tra. Kết quả sau sửa xem bảng dưới.

**Các thử nghiệm agent trên dev (N = 16 câu có đáp án; "đúng" = đã trả lời và citation trỏ đúng chunk evidence, q08/q16 và các câu thay đổi được đọc lại thủ công):**

| Thí nghiệm | Cấu hình | Trả lời đúng | Từ chối nhầm | Lượt tìm (cả 18 câu) | Context TB (ký tự) | ms (trung vị) |
|---|---|---|---|---|---|---|
| Pipeline gốc | Dense top-3, budget 2000, prompt cũ | 14/16 | 2 (q08, q16) | 26 | 1,536 | 2,439 |
| E1 | Dense top-8, budget 8000, prompt cũ | 14/16 | q16, q02 (lỗi định dạng) | 20 | — | 2,406 |
| E2 | Hybrid top-10, budget 10000, prompt cũ | 14/16 | q01, q02 (lỗi định dạng) | 18 | — | 2,493 |
| E3 | Dense top-3, cấm lặp query, prompt cũ | 13/16 | q14, q16 + q08 trả lời sai | 26 | — | 2,385 |
| F0 | Dense top-3, budget 2000, prompt đã sửa citation | 13/16 | 3 (q08, q14, q16) | 28 | 1,564 | 2,201 |
| F1 | Dense top-8, budget 8000, prompt đã sửa | 15/16 | 1 (q16) | 24 | 7,514 | 3,015 |
| **F2 (chọn)** | **Hybrid top-10, budget 10000, prompt đã sửa** | **16/16** | **0** | **18** | 9,221 | 3,070 |

(Cột ms là thời gian trung vị mỗi câu; "—" là chưa tính cho thí nghiệm đó.)

**Tiêu chí chọn cấu hình (đặt trước khi chạy F0–F2):** nhiều câu dev trả lời đúng có citation đúng nhất; nếu hòa thì chọn cấu hình ít lượt tìm và context ngắn hơn. F2 thắng ở tiêu chí đầu (16/16) và có ít lượt tìm nhất (18, bằng E2). Đánh đổi: context trung bình dài gấp ~6 lần (9.2k so với 1.5k ký tự) và độ trễ tăng từ ~2.4 s lên ~3.1 s mỗi câu, đổi lại không còn từ chối nhầm.

**Lưu ý về nhiễu:** F0 chỉ khác pipeline gốc ở prompt citation nhưng cho 13/16 thay vì 14/16 (q14 mất do hết lượt tìm). Cùng retrieval, kết quả vẫn lệch 1 câu, nên chênh lệch 1 câu giữa các thí nghiệm nằm trong khoảng nhiễu của LLM với N = 16; kết luận chắc chắn nhất là các sửa chữa q08/q16 và định dạng citation, vì có nguyên nhân cơ chế được xác nhận bằng trace.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context, xử lý lỗi:** một tool `search(query)` do Gemini gọi; retriever cố định theo cấu hình (cấu hình cuối là hybrid RRF của dense pgvector và TF-IDF). Tối đa 3 lượt tìm; dừng khi (a) model trả lời và mọi citation có trong context và xuất hiện dạng `[id]`, (b) model nói không đủ bằng chứng hoặc citation không hợp lệ → từ chối, (c) hết lượt tìm. Context: chunk trùng được bỏ, chọn theo thứ tự xếp hạng tới khi hết budget, chunk không vừa budget bị bỏ cả khối, chỉ gửi context tích lũy mới nhất. Lỗi API 429 được thử lại tối đa 3 lần; lỗi khác dừng và được ghi vào trace; không có kết quả giả.

| Cấu hình / split | Tool use | Số lượt gọi (tổng) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Pipeline gốc · dev | 18/18 | 26 | 2 / 0 / 14, 1.75 | 14/14 | 14/14 | 2/2 | 2/16 |
| Pipeline gốc · test | 8/8 | 10 | 0 / 0 / 6, 2.00 | 6/6 | 6/6 | 2/2 | 0/6 |
| **Cấu hình chọn (F2) · dev** | 18/18 | 18 | 0 / 0 / 16, **2.00** | 19/19 | 16/16 | 2/2 | **0/16** |
| **Cấu hình chọn (F2) · test** | 8/8 | 10 | 0 / 0 / 6, **2.00** | 9/9 | 6/6 | 2/2 | 0/6 |

Tỷ lệ citation hợp lệ chỉ kiểm tra ID có trong context; việc chunk có hỗ trợ câu trả lời được kiểm tra thủ công.

Test đã được chạy **hai lần**: lần 1 với pipeline gốc (trước khi cải thiện, 6/6 trả lời và 2/2 từ chối đúng), lần 2 một lần duy nhất với cấu hình đã chốt trên dev (F2). Không có thay đổi nào được thực hiện dựa trên kết quả test; cấu hình chọn theo tiêu chí công bố trước, chỉ dùng dev.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q16 | Có đáp án, evidence trong bảng (lỗi dev) | Gốc: 3 lần `search("IFC net income fiscal 2026")`, top-3 giống hệt, không có trang 33 → F2: 1 lần, hybrid trả về chunk `p33:para:000` | Gốc: "Không đủ bằng chứng" (từ chối nhầm, 0). F2: "IFC's net income was $2,038 million [wbg-ar-2026:p33:para:000]" | Gốc 0; F2: 2, citation hỗ trợ (dòng bảng Net Income 2,038). Dense không tìm được, lexical bắt được từ khóa |
| q19 | Có đáp án, test | F2: 1 lần `search`, context chứa `p12:para:001` và `p12:para:002` | "PHIT is expected to create, protect, and safeguard 16,000 health and health-support jobs ... [p12:para:001] [p12:para:002]" | 2; khớp nguyên văn đáp án (lần chạy gốc cho 2 nhưng context chỉ được 1 điểm vì thiếu câu "create, protect") |
| u04 | Thiếu bằng chứng, test | F2: 3 lần `search` không tìm thấy nội dung về xe điện/đội xe | "Không đủ bằng chứng để trả lời." | Abstention đúng; đã quét 255 chunk, không có nội dung liên quan |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian, chi phí:**
  - Pipeline gốc: `starter/results/agent-dev/` và `agent-test/` (traces, reviews, summary).
  - Thí nghiệm dev: `starter/results/agent-dev-E1-dense-k8/`, `-E2-hybrid-k10/`, `-E3-norepeat/`, `-F0-dense-k3/`, `-F1-dense-k8/`, `-F2-hybrid-k10/`.
  - Cấu hình chọn trên test: `starter/results/agent-test-F2-hybrid-k10/` (có `traces.json`, `reviews.json`, `summary.json`).
  - Retrieval: `starter/results/benchmark-dev-baseline`, `-budget2000`, `-topk5`, `benchmark-dev-k10-b10000`, `benchmark-test-baseline`, `-chosen`, `benchmark-test-k10-b10000`.
  - Context 0/1/2: `starter/results/context-review-test.json`.
  - Lệnh chạy cấu hình cuối: `python starter/run_agent_eval.py --cases starter/data/converted/cases.json --split test --max-searches 3 --retriever hybrid --top-k 10 --budget 10000 --output starter/results/agent-test-F2-hybrid-k10`, rồi `python starter/agent_metrics.py --cases ... --traces ... --reviews ... --split test --output ...`.
  - Thời gian trung vị mỗi câu: ~2.4 s (pipeline gốc), ~3.1 s (F2); mỗi câu mất thêm 3.5 s nghỉ giữa các lần gọi. Retrieval cục bộ chỉ ~40–90 ms. **Chi phí Gemini không được ghi nhận** (trace không lưu số token); embedding và vector store local nên 0 USD.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:** từ chối nhầm trên dev giảm từ 2/16 xuống 0/16; context evidence trên test từ 0/6 (baseline) lên 6/6; grounded trung bình dev từ 1.75 lên 2.00. Đánh đổi là context dài hơn ~6 lần và chậm hơn ~0.7 s mỗi câu.

## Giới hạn

1. **Nhãn do Claude chấm**, chưa có người rà soát độc lập; đề yêu cầu người chấm xác nhận. Các nhãn nên được kiểm tra lại, đặc biệt q08, q16, q19 và các câu thiếu bằng chứng.
2. **Cỡ mẫu nhỏ** (16 dev, 6 test có đáp án): chênh lệch 1 câu nằm trong khoảng nhiễu của LLM (F0 và pipeline gốc chỉ khác prompt nhưng lệch 1 câu). Trên test, ba phương pháp hòa nhau, nên không kết luận được hybrid vượt trội dense nói chung.
3. **Cấu hình chọn đã được điều chỉnh sau khi phân tích hai lỗi dev cụ thể** (q08, q16); dev có thể bị "học thuộc". Test chỉ được dùng một lần cho cấu hình cuối, nhưng test đã được chạy cho pipeline gốc trước đó, nên đây không phải một lần chạy test duy nhất.
4. **Không dùng filter metadata** dù 6 câu có trường `filter`; chưa đo tác động của filter lên agent.
5. **Chi phí token Gemini không được ghi**; không tính được chi phí tiền tệ cho agent.
6. **Một tài liệu duy nhất, tiếng Anh**; bảng số liệu chỉ được xử lý như text (pypdf). q16 được cứu nhờ lexical, không nhờ phân tích bảng.
7. **Agent chỉ dùng một tool tìm kiếm với retriever cố định**; chưa có rerank, viết lại câu hỏi do hệ thống (ngoài query do model tự viết) hoặc multi-hop thực sự (mọi câu đều đạt chỉ với 1 lượt tìm, trừ các câu từ chối).
8. **Môi trường:** test đơn vị của khóa học cần `PYTHONUTF8=1` trên Windows (một test đọc file không chỉ định encoding); cổng DB là 55600 thay vì 55433. Với `PYTHONUTF8=1`, 15/15 test đạt.
