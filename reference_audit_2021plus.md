# Reference Audit (2021+)

Audit date: 2026-03-20

Scope:
- Paper: `C:\Users\LENOVO\Downloads\res\latex\samplepaper.tex`
- Bibliography: `C:\Users\LENOVO\Downloads\res\latex\references.bib`

Verification method:
- Direct automated access to Google Scholar was blocked in this environment.
- Each cited title was therefore checked by exact-title search and matched against an official source such as ACL Anthology, publisher page, DOI landing page, ScienceDirect, Springer, MIT Press, IEEE Xplore, or arXiv.
- Citation placement was then reviewed against the sentence where the citation appears in `samplepaper.tex`.

Summary:
- Active cited references: 21
- All 21 cited references are real publications or real preprints.
- All 21 are from 2021 onward.
- Most citation placements are appropriate.
- A few placements are real but somewhat indirect or broader than the claim they support.

## Strong and well-placed references

1. `lin2022laoplm`
- Title: `LaoPLM: Pre-trained Language Models for Lao`
- Official source: https://aclanthology.org/2022.lrec-1.698/
- Placement: appropriate for Lao NLP resources, Lao-specific model support, and Lao language coverage.

2. `nguyen2025semi`
- Title: `Semi-Automatic Construction and Benchmarking of a Word-Segmented Corpus for Lao Using LLMs and Transformer Models`
- Official source: https://informatica.si/index.php/informatica/article/view/11195
- Placement: appropriate for Lao resource scarcity, segmentation difficulty, and continuous-script preprocessing issues.

3. `lamin2025crosslingual`
- Title: `Cross-Lingual Sentiment Analysis in Low-Resource Languages: A Recent Review on Tasks, Methods and Challenges`
- Official source: https://thesai.org/Publications/ViewPaper?Code=IJACSA&Issue=11&SerialNo=44&Volume=16
- Placement: appropriate for low-resource sentiment analysis, multilingual transfer, and review-level sentiment discussion.

4. `tan2023survey`
- Title: `A Survey of Sentiment Analysis: Approaches, Datasets, and Future Research`
- Official source: https://www.mdpi.com/2076-3417/13/7/4550
- Placement: appropriate for general sentiment-analysis background and metric discussion.

5. `alahmadi2025generalizing`
- Title: `Generalizing Sentiment Analysis: A Review of Progress, Challenges, and Emerging Directions`
- Official source: https://link.springer.com/article/10.1007/s13278-025-01461-8
- Placement: appropriate for domain shift, robustness, and limits of traditional approaches.

6. `zhang2022absa`
- Title: `A Survey on Aspect-Based Sentiment Analysis: Tasks, Methods, and Challenges`
- Official source: https://ieeexplore.ieee.org/document/9996141
- Placement: appropriate in the related-work paragraph contrasting review-level sentiment analysis and ABSA.

7. `chebolu2023absa_datasets`
- Title: `A Review of Datasets for Aspect-Based Sentiment Analysis`
- Official source: https://aclanthology.org/2023.ijcnlp-main.41/
- Placement: appropriate as supporting background for ABSA datasets.

8. `mahmood2025application_domains`
- Title: `Application Domains of Aspect and Sentiment Classification Techniques: A Survey`
- Official source: https://www.sciencedirect.com/science/article/pii/S0925231224020083
- Placement: appropriate for application-domain framing in related work.

9. `smid2025crosslingual_absa`
- Title: `Cross-Lingual Aspect-Based Sentiment Analysis: A Survey on Tasks, Approaches, and Challenges`
- Official source: https://www.sciencedirect.com/science/article/pii/S1566253525001460
- Placement: appropriate as ABSA and cross-lingual sentiment background.

10. `wu2025mabsa`
- Title: `M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis`
- Official source: https://aclanthology.org/2025.emnlp-main.128/
- Placement: appropriate as multilingual ABSA dataset background.

11. `hu2022lora`
- Title: `LoRA: Low-Rank Adaptation of Large Language Models`
- Official source: https://arxiv.org/abs/2106.09685
- Placement: appropriate for the LoRA method description.

12. `mao2025lora_survey`
- Title: `A Survey on LoRA of Large Language Models`
- Official source: https://journal.hep.com.cn/fcs/EN/10.1007/s11704-024-40663-9
- Placement: appropriate for LoRA background and broader PEFT context.

13. `nwaiwu2025peft_lowresource`
- Title: `Parameter-Efficient Fine-Tuning for Low-Resource Text Classification: A Comparative Study of LoRA, IA3, and ReFT`
- Official source: https://www.frontiersin.org/articles/10.3389/fdata.2025.1677331
- Placement: appropriate for low-resource PEFT comparison and efficiency framing.

14. `opitz2024metrics`
- Title: `A Closer Look at Classification Evaluation Metrics and a Critical Reflection of Common Evaluation Practice`
- Official source: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00675/122720/A-Closer-Look-at-Classification-Evaluation-Metrics
- Placement: appropriate for evaluation metrics and macro-F1 justification.

## Real references, but citation placement is somewhat indirect

15. `subramanian2025sentiment`
- Title: `KEC_TECH_TITANS@DravidianLangTech 2025: Sentiment Analysis for Low-Resource Languages: Insights from Tamil and Tulu using Deep Learning and Machine Learning Models`
- Official source: https://aclanthology.org/2025.dravidianlangtech-1.48/
- Status: real and relevant to low-resource sentiment analysis.
- Placement note: acceptable in comparative discussion, but it is an individual system paper, not a general survey.

16. `mystic2025hybrid`
- Title: `MysticCIOL@DravidianLangTech 2025: A Hybrid Framework for Sentiment Analysis in Tamil and Tulu Using Fine-Tuned SBERT Embeddings and Custom MLP Architectures`
- Official source: https://aclanthology.org/2025.dravidianlangtech-1.28/
- Status: real and relevant to low-resource sentiment analysis.
- Placement note: acceptable as supporting evidence for low-resource sentiment results, but it is still a shared-task system paper rather than a broad survey.

17. `durairaj2025overview`
- Title: `Overview of the Shared Task on Sentiment Analysis in Tamil and Tulu`
- Official source: https://aclanthology.org/2025.dravidianlangtech-1.124.pdf
- Status: real and relevant.
- Placement note: strong as shared-task overview evidence; better than the system papers when making broader claims.

18. `ruder2021xtremer`
- Title: `XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation`
- Official source: https://arxiv.org/abs/2104.07412
- Status: real and relevant to multilingual evaluation.
- Placement note: acceptable for multilingual transfer background, but it is not a survey of sentiment analysis.

19. `han2021pretrained`
- Title: `Pre-trained Models: Past, Present and Future`
- Official source: https://www.sciencedirect.com/science/article/pii/S2666651021000231
- Status: real and relevant to pretrained-model background.
- Placement note: appropriate for PLM background, but broader than sentiment analysis.

20. `min2021plm_survey`
- Title: `Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey`
- Official source: https://arxiv.org/abs/2111.01243
- Status: real and relevant to PLM background.
- Placement note: appropriate for PLM methodology context, but broader than sentiment analysis.

21. `zhang2024asean_alignment`
- Title: `Cross-Lingual Word Alignment for ASEAN Languages with Contrastive Learning`
- Official source: https://arxiv.org/abs/2407.05054
- Status: real and Lao-relevant.
- Placement note: this is the weakest topical match. It supports Lao/ASEAN cross-lingual representation learning, but it is not specifically about sentiment analysis.

## Placement review inside the paper

Good placements:
- `lin2022laoplm`, `nguyen2025semi` in the Lao resource and preprocessing discussion.
- `zhang2022absa`, `chebolu2023absa_datasets`, `smid2025crosslingual_absa`, `wu2025mabsa` in Related Work.
- `hu2022lora`, `mao2025lora_survey`, `nwaiwu2025peft_lowresource` in Fine-Tuning Strategies.
- `opitz2024metrics`, `tan2023survey` in Evaluation Metrics.

Placements worth tightening if the paper is revised again:
- Line 98 area: `lamin2025crosslingual` is used to support the business importance of sentiment analysis in digital-service platforms. This is not wrong, but `tan2023survey` would be a more direct citation for that sentence.
- Line 102 area: `nguyen2025semi` is included in a sentence about the limitations of classical machine learning and rule-based approaches. It is Lao-relevant but not the strongest direct support for that claim.
- Line 127 area: `ruder2021xtremer` and `han2021pretrained` support transfer-learning background well, but they are not sentiment-specific surveys.
- Line 127 area: `zhang2024asean_alignment` is regionally relevant but still tangential to sentiment analysis.

Final judgment:
- The current 21 references are real.
- The current 21 references are all 2021+.
- The bibliography is now cleaner and more focused than before.
- Most citations are placed reasonably.
- The paper is safe on reference authenticity.
- The only remaining weakness is not fake or wrong citation, but a small number of broad or indirect placements that could be tightened in a final polish pass.
