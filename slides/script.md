# Speaker Script -- Gradient Inversion Attacks in Federated Learning

Bilingual (EN/VI) speaking notes, one entry per slide, in the same order as `slide.tex`.

## Slide 1: Title
**EN:** Good morning everyone. Today I'll present our project on gradient inversion attacks in Federated Learning -- how a shared gradient can be turned back into the private image and label that produced it. We'll walk through four attack methods, one defense, and the experiments that compare them.
**VI:** Chào mọi người, hôm nay em xin trình bày đề tài về tấn công gradient inversion trong Federated Learning -- tức là làm thế nào để từ một gradient được chia sẻ, kẻ tấn công có thể khôi phục lại ảnh và nhãn riêng tư ban đầu. Bọn em sẽ đi qua bốn phương pháp tấn công, một cơ chế phòng thủ, và các thí nghiệm so sánh chúng.

## Slide 2: Outline
**EN:** Here's the roadmap: motivation and threat model first, then the four attacks -- DLG, iDLG, Inverting Gradients, and R-GAP -- then the DP-SGD defense, our experimental evaluation across four research questions, and finally the key takeaways.
**VI:** Đây là dàn ý: đầu tiên là động lực và mô hình đe dọa, tiếp theo là bốn phương pháp tấn công -- DLG, iDLG, Inverting Gradients và R-GAP -- sau đó là cơ chế phòng thủ DP-SGD, phần đánh giá thực nghiệm với bốn câu hỏi nghiên cứu, và cuối cùng là các kết luận chính.

## Slide 3: Outline (Motivation & Threat Model section)
**EN:** Let's start with why this matters -- what Federated Learning promises, and what an attacker is actually able to see.
**VI:** Bắt đầu với phần tại sao vấn đề này quan trọng -- Federated Learning hứa hẹn điều gì, và kẻ tấn công thực sự có thể quan sát được những gì.

## Slide 4: Federated Learning's Privacy Promise
**EN:** In Federated Learning, each client trains locally on its own private data and only sends back a gradient or weight update; the server just averages these updates, as shown in the FedAvg rule here. The whole appeal is that raw data supposedly never leaves the device -- only this aggregated update is shared.
**VI:** Trong Federated Learning, mỗi client huấn luyện cục bộ trên dữ liệu riêng của mình và chỉ gửi về một gradient hoặc bản cập nhật trọng số; server chỉ việc gộp các cập nhật này lại theo công thức FedAvg như trên slide. Điểm hấp dẫn là dữ liệu gốc được cho là không bao giờ rời khỏi thiết bị -- chỉ có bản cập nhật đã tổng hợp này được chia sẻ.

## Slide 5: The Gradient Inversion Problem
**EN:** We study an honest-but-curious attacker -- either the server itself or someone eavesdropping on the channel -- who follows the protocol exactly but tries to extract private data from what it legitimately sees: the architecture, the current weights, and the leaked gradient. The question is whether that gradient alone is enough to recover the private batch.
**VI:** Bọn em xét một kẻ tấn công kiểu "honest-but-curious" -- có thể là chính server hoặc một kẻ nghe lén trên đường truyền -- tuân thủ đúng giao thức nhưng cố gắng trích xuất dữ liệu riêng tư từ những gì nó quan sát được một cách hợp lệ: kiến trúc mô hình, trọng số hiện tại, và gradient bị lộ. Câu hỏi đặt ra là liệu chỉ riêng gradient đó có đủ để khôi phục lại batch dữ liệu riêng tư hay không.

## Slide 6: Outline (Attack Methods section)
**EN:** Now the four attacks themselves, starting with the original Deep Leakage from Gradients method.
**VI:** Bây giờ là bốn phương pháp tấn công, bắt đầu từ phương pháp gốc Deep Leakage from Gradients.

## Slide 7: DLG
**EN:** DLG starts from a random dummy image and label, and optimizes both by gradient descent so that their gradient matches the leaked one under an L2 distance. Because this objective is itself a gradient, DLG needs to differentiate through it a second time -- what's called double backpropagation. Optimizing the label jointly with the image makes it slow and sometimes unstable.
**VI:** DLG bắt đầu từ một ảnh và nhãn giả ngẫu nhiên, rồi tối ưu cả hai bằng gradient descent sao cho gradient của chúng khớp với gradient bị lộ theo khoảng cách L2. Vì hàm mục tiêu này bản thân là một gradient, nên DLG phải đạo hàm qua nó một lần nữa -- gọi là "double backpropagation". Việc tối ưu đồng thời cả nhãn lẫn ảnh khiến thuật toán chạy chậm và đôi khi không ổn định.

## Slide 8: iDLG
**EN:** iDLG's insight is that for cross-entropy loss, the sign of the last layer's gradient directly tells you the true label -- no optimization needed for that part. So only the image is optimized, which converges faster and more reliably than DLG.
**VI:** Ý tưởng của iDLG là với hàm mất mát cross-entropy, dấu của gradient ở lớp cuối cùng cho biết trực tiếp nhãn thật -- không cần tối ưu phần này. Vì vậy chỉ còn ảnh cần tối ưu, giúp quá trình hội tụ nhanh và ổn định hơn so với DLG.

## Slide 9: Inverting Gradients
**EN:** Inverting Gradients replaces the L2 distance with a cosine-similarity cost, because most of the useful information is in the gradient's direction, not its magnitude. It adds a total-variation prior to keep images smooth, and uses sign-based Adam updates. These choices are what let it scale all the way up to ImageNet-sized ResNet-18.
**VI:** Inverting Gradients thay khoảng cách L2 bằng chi phí cosine similarity, vì phần lớn thông tin hữu ích nằm ở hướng của gradient chứ không phải độ lớn. Nó thêm một prior total-variation để giữ ảnh mượt, và dùng cập nhật Adam theo dấu. Chính những lựa chọn này giúp phương pháp mở rộng được đến ResNet-18 ở quy mô ImageNet.

## Slide 10: R-GAP
**EN:** R-GAP takes a completely different route: no optimization at all. It treats the per-layer gradient equation as a linear system and solves it directly, recursing from the output layer back to the input. When the rank conditions hold, it recovers the exact input -- not an approximation.
**VI:** R-GAP đi theo một hướng hoàn toàn khác: không hề có vòng lặp tối ưu. Nó coi phương trình gradient ở mỗi lớp là một hệ phương trình tuyến tính và giải trực tiếp, đệ quy từ lớp đầu ra ngược về lớp đầu vào. Khi điều kiện về hạng ma trận được thỏa mãn, nó khôi phục chính xác đầu vào -- không phải một xấp xỉ.

## Slide 11: Outline (DP-SGD Defense section)
**EN:** Given these attacks, what can a defender actually do? Let's look at DP-SGD.
**VI:** Với những tấn công này, phía phòng thủ có thể làm gì? Hãy xem xét DP-SGD.

## Slide 12: DP-SGD as a Post-Hoc Defense
**EN:** We reuse DP-SGD's clip-and-noise mechanics here, but applied once to the single, already-aggregated gradient the attacker sees -- not the per-example, training-integrated version that gives DP-SGD its formal privacy guarantee. Clip it to a norm C, then add Gaussian noise scaled by sigma, before the attacker ever sees the gradient, and we sweep the same (C, sigma) budget across every attack so the comparison is fair.
**VI:** Ở đây bọn em chỉ tái sử dụng cơ chế clip-and-noise của DP-SGD, áp dụng một lần lên đúng một gradient đã được gộp mà kẻ tấn công quan sát được -- không phải phiên bản per-example, tích hợp vào quá trình huấn luyện, vốn mới là thứ mang lại bảo đảm differential privacy chính thức của DP-SGD. Cắt (clip) nó về một ngưỡng chuẩn C, rồi cộng thêm nhiễu Gaussian với hệ số sigma, trước khi kẻ tấn công kịp nhìn thấy gradient, và bọn em quét cùng một ngân sách (C, sigma) cho tất cả các tấn công để việc so sánh công bằng.

## Slide 13: Outline (Experimental Evaluation section)
**EN:** Now let's see how all of this plays out experimentally, across four research questions.
**VI:** Bây giờ hãy xem tất cả những điều này thể hiện ra sao trong thực nghiệm, qua bốn câu hỏi nghiên cứu.

## Slide 14: Evaluation: Four Research Questions
**EN:** We ask four questions. First, under the exact same architecture and batch size, how does an optimization-based attack compare to a closed-form one? Second, does Inverting Gradients -- built for scale -- still work on a much deeper ResNet-18? Third, how much does each attack degrade when the gradient comes from a batch of two images instead of one? And fourth, can DP-SGD's clipping and noise actually suppress these attacks, or does the noise needed vary a lot between them?
**VI:** Bọn em đặt ra bốn câu hỏi. Thứ nhất, trên cùng kiến trúc và batch size, tấn công dựa trên tối ưu so với tấn công dạng closed-form thì ai hơn? Thứ hai, liệu Inverting Gradients -- vốn được thiết kế cho quy mô lớn -- có còn hoạt động tốt trên ResNet-18 sâu hơn nhiều không? Thứ ba, mỗi tấn công suy giảm bao nhiêu khi gradient đến từ một batch hai ảnh thay vì một? Và thứ tư, liệu DP-SGD có thực sự ngăn chặn được các tấn công này không, hay mức nhiễu cần thiết lại khác nhau rất nhiều giữa chúng?

## Slide 15: Evaluation: Victim Configurations
**EN:** Here's every configuration we actually ran. The key detail is the LeNet row shared by DLG, iDLG, and Inverting Gradients -- same architecture, same initial weights, same image -- which is what makes RQ1 and RQ4 a fair, matched comparison instead of an apples-to-oranges one.
**VI:** Đây là toàn bộ các cấu hình bọn em đã chạy thực tế. Chi tiết quan trọng là dòng LeNet dùng chung cho DLG, iDLG và Inverting Gradients -- cùng kiến trúc, cùng trọng số khởi tạo, cùng ảnh -- đây chính là điều giúp RQ1 và RQ4 trở thành một so sánh công bằng, chứ không phải khập khiễng.

## Slide 16: RQ1: Baseline Comparison, Shared LeNet
**EN:** On the stored LeNet run, DLG and iDLG finish close but slightly noisy, Inverting Gradients trails them at 26.48 dB, and R-GAP matches the image in float64. The infinite PSNR in the table appears after the result is cast to float32; before that cast the error is about 10 to the minus 22. A GPU rerun reproduced the Inverting Gradients number and the labels, but not the exact DLG L-BFGS path.
**VI:** Trên lần chạy LeNet đã lưu, DLG và iDLG kết thúc gần ảnh gốc nhưng còn nhiễu nhẹ, Inverting Gradients thấp hơn ở 26,48 dB, và R-GAP khớp ảnh ở float64. PSNR vô hạn trong bảng xuất hiện sau khi ép kết quả về float32; trước lần ép đó sai số khoảng 10 mũ âm 22. Lần chạy GPU sau này lặp lại được số của Inverting Gradients và nhãn, nhưng không đi lại đúng quỹ đạo L-BFGS của DLG.

## Slide 17: RQ1: Convergence of DLG vs. iDLG
**EN:** This figure tracks ten snapshots across the 300-iteration optimization. Both start from pure noise, but iDLG -- since it only has to optimize the image, not the label -- settles into the right shape noticeably earlier than DLG.
**VI:** Hình này ghi lại mười mốc trong quá trình tối ưu 300 vòng lặp. Cả hai đều bắt đầu từ nhiễu ngẫu nhiên, nhưng iDLG -- vì chỉ phải tối ưu ảnh chứ không phải nhãn -- định hình đúng hình dạng sớm hơn rõ rệt so với DLG.

## Slide 18: RQ1: R-GAP Needs Float64
**EN:** R-GAP's stored exact result depends on float64, and the infinite PSNR appears only after the reconstruction is cast to float32. Repeating the attack in float32 collapses to about minus 20 dB at the last linear solve. A GPU rerun reproduced that collapse at minus 19.59 dB.
**VI:** Kết quả chính xác đã lưu của R-GAP phụ thuộc vào float64, và PSNR vô hạn chỉ xuất hiện sau khi ép kết quả về float32. Chạy lại ở float32 thì sụp xuống khoảng âm 20 dB ở phép giải tuyến tính cuối. Lần chạy GPU lặp lại đúng sự sụp đổ này, ở âm 19,59 dB.

## Slide 19: RQ2: Scaling to ResNet-18
**EN:** Here Inverting Gradients attacks an untrained ResNet-18 with 11.7 million parameters and a 224-by-224 image. The 17.33 dB figure is peak-1 PSNR after denormalization, paired with MSE 0.0185. The error on the normalized tensor is about 0.30, so those two MSE numbers must not be read as the same quantity. The image is still identifiable -- pose and fur color -- but well below the 26.48 dB LeNet result.
**VI:** Ở đây Inverting Gradients tấn công một ResNet-18 chưa huấn luyện với 11,7 triệu tham số và ảnh 224x224. Con số 17,33 dB là PSNR đỉnh 1 sau khi denormalize, đi cùng MSE 0,0185. Sai số trên tensor đã normalize khoảng 0,30, nên hai giá trị MSE này không phải cùng một đại lượng. Ảnh vẫn nhận diện được -- tư thế và màu lông -- nhưng thấp hơn rõ so với 26,48 dB của LeNet.

## Slide 20: RQ3: Batch Size 1 vs. 2 -- Optimization-Based Attacks
**EN:** Moving from one image to a batch of two hurts differently depending on the attack. DLG and iDLG on LeNet lose about 11 to 12 dB, but Inverting Gradients on ResNet-18 barely drops at all -- less than 1 dB -- simply because its batch-1 result wasn't that precise to begin with, so there's less to lose.
**VI:** Việc chuyển từ một ảnh sang batch hai ảnh ảnh hưởng khác nhau tùy từng tấn công. DLG và iDLG trên LeNet mất khoảng 11 đến 12 dB, nhưng Inverting Gradients trên ResNet-18 hầu như không giảm -- chưa đến 1 dB -- đơn giản vì kết quả ở batch 1 vốn đã không quá chính xác, nên không còn nhiều để mất.

## Slide 21: RQ3: Inverting Gradients, ResNet-18, Batch Size 2
**EN:** Here are the actual batch-2 reconstructions on ResNet-18. Even with that PSNR drop, both images are still individually recognizable -- the attack degrades gracefully rather than collapsing outright.
**VI:** Đây là ảnh khôi phục thực tế ở batch 2 trên ResNet-18. Dù PSNR giảm, cả hai ảnh vẫn có thể nhận diện riêng biệt -- tấn công suy giảm từ từ chứ không sụp đổ hoàn toàn.

## Slide 22: RQ3: R-GAP at Batch Size 2 -- the Linear-Combination Property
**EN:** R-GAP behaves very differently depending on architecture. FCN3 has no convolutional layers, so even though it does have a LeakyReLU at every layer, R-GAP inverts that exactly -- what actually matters is the absence of convolution, not linearity -- and the result closely matches the direction of the pixel average of the two images. On the convolutional LeNetRGAP, the extra forward-pass constraints break that, and the result matches neither image nor their average -- it's essentially noise.
**VI:** R-GAP hành xử rất khác nhau tùy kiến trúc. FCN3 không có lớp tích chập nào, nên dù mỗi layer đều có LeakyReLU, R-GAP vẫn đảo ngược được chính xác hàm đó -- điều thực sự quan trọng là không có tích chập, chứ không phải tính tuyến tính -- và kết quả khớp rất gần về hướng với trung bình pixel của hai ảnh. Nhưng trên LeNetRGAP có tích chập, các ràng buộc forward-pass bổ sung phá vỡ điều đó, và kết quả không khớp với ảnh nào cũng không khớp với trung bình -- về cơ bản là nhiễu.

## Slide 23: RQ3: R-GAP Batch-2 Reconstructions
**EN:** Seeing it confirms the numbers: the CNN variant on top is visually just colored noise, while FCN3 on the bottom is a clean, recognizable blend of the horse and the ostrich-like bird.
**VI:** Nhìn hình sẽ thấy rõ con số đó đúng: phiên bản CNN ở trên chỉ là nhiễu màu sắc, trong khi FCN3 ở dưới là một bản pha trộn rõ ràng, nhận diện được giữa con ngựa và con chim giống đà điểu.

## Slide 24: RQ4: DP-SGD vs. DLG Family (LeNet)
**EN:** On the same LeNet, clipping alone already separates the two cost functions: iDLG's L2 cost collapses under a half-norm clip even with zero noise, while Inverting Gradients' cosine cost does not. The 27.94 dB defense entry is higher than the 26.48 dB baseline because that defense run uses 4,800 median-filtered steps. A paired ablation scored 27.85 dB at 4,800 steps and 26.36 dB at 24,000, so the extra PSNR is the budget, not a benefit of the defense.
**VI:** Trên cùng một LeNet, chỉ riêng clipping đã tách hai hàm chi phí: chi phí L2 của iDLG sụp khi clip còn một nửa norm dù nhiễu bằng 0, còn chi phí cosine của Inverting Gradients thì không. Mức 27,94 dB của defense cao hơn baseline 26,48 dB vì lượt defense chỉ chạy 4.800 bước có median filter. Thí nghiệm đối chứng cho 27,85 dB ở 4.800 bước và 26,36 dB ở 24.000 bước, nên phần PSNR cao hơn đến từ số bước, không phải từ phòng thủ.

## Slide 25: RQ4: Reconstructions Under DP-SGD (LeNet)
**EN:** You can see it directly here: iDLG on top turns to colored noise well before Inverting Gradients on the bottom does, at the exact same noise levels.
**VI:** Có thể thấy trực tiếp ở đây: iDLG ở trên biến thành nhiễu màu sớm hơn nhiều so với Inverting Gradients ở dưới, dù cùng một mức nhiễu.

## Slide 26: RQ4: DP-SGD vs. R-GAP
**EN:** R-GAP tells the opposite story. It's exact with no noise, and still holds up at 31 dB with a tiny noise multiplier, but one more order of magnitude of noise completely collapses it to negative PSNR. That breaking point is three to four orders of magnitude below where iDLG breaks -- so a noise level too small to even touch DLG or iDLG is already enough to defeat R-GAP outright.
**VI:** R-GAP lại kể một câu chuyện ngược lại. Nó chính xác tuyệt đối khi không có nhiễu, vẫn giữ được 31 dB với một mức nhiễu rất nhỏ, nhưng chỉ cần tăng thêm một bậc nhiễu là nó sụp đổ hoàn toàn về PSNR âm. Điểm sụp đổ này thấp hơn điểm sụp đổ của iDLG đến ba, bốn bậc độ lớn -- nghĩa là một mức nhiễu quá nhỏ để ảnh hưởng đến DLG hay iDLG đã đủ để đánh bại hoàn toàn R-GAP.

## Slide 27: RQ4: R-GAP Reconstructions Under DP-SGD
**EN:** The break is visible right here, between these two noise columns -- a sharp cliff, not a gradual fade like the other attacks show.
**VI:** Điểm sụp đổ hiện rõ ngay ở đây, giữa hai cột mức nhiễu này -- một vách dựng đứng, không phải suy giảm từ từ như các tấn công khác.

## Slide 28: Outline (Conclusion section)
**EN:** Let's wrap up with what all of this means.
**VI:** Hãy tổng kết lại ý nghĩa của tất cả những điều này.

## Slide 29: Key Takeaways
**EN:** A single leaked gradient is enough to reconstruct private images and labels. R-GAP matches the image in float64, but that exactness is fragile to precision and to batch size, and wall-clock times from different devices are not a ranking. Optimization-based attacks still work on ResNet-18, at lower fidelity. No single DP-SGD budget stops every attack family: noise that defeats the stored R-GAP run leaves DLG and iDLG recognizable.
**VI:** Chỉ một gradient bị lộ cũng đủ để khôi phục ảnh và nhãn. R-GAP khớp ảnh ở float64, nhưng độ chính xác đó mong manh trước sai số số học và kích thước batch, và thời gian chạy trên các máy khác nhau không phải một bảng xếp hạng. Tấn công dựa trên tối ưu vẫn chạy được trên ResNet-18, với độ trung thực thấp hơn. Không có một mức DP-SGD nào chặn đều mọi họ tấn công: mức nhiễu đánh bại lần chạy R-GAP đã lưu vẫn để DLG và iDLG nhận ra được.

## Slide 30: Thank You / Q&A
**EN:** Thank you for listening -- I'm happy to take any questions.
**VI:** Em xin cảm ơn mọi người đã lắng nghe -- em sẵn sàng giải đáp các câu hỏi.
