# Phase 3.5 Claim–Evidence Matrix

감사일: 2026-08-23
규칙: 각 행은 하나의 검증 가능한 주장과 하나의 주 evidence class만 가진다. 서로 다른 종류의 근거가 필요한 복합 주장은 별도 행으로 분리했다. `supporting source or file`의 논문 metadata와 URL은 `references.bib`에 수록했다.

| claim | evidence class | supporting source or file | what the evidence supports | what it does not support | required experiment or action |
|---|---|---|---|---|---|
| 복구 설정은 2022–2024 월별 `ODIAC2025` 1 km `excl_intl` GeoTIFF를 지시한다 | P | recovered archive `configs/delhi_2022_2024.yaml` | 의도된 파일명 패턴·기간 | 해당 파일이 실제 NPZ 생성에 사용되었다는 사실 | 36개 source file manifest와 checksum 복구 |
| 현재 NPZ를 만든 정확한 ODIAC release는 확정되지 않았다 | U | project에 source TIFF/header/download manifest 부재; `IMPLEMENTATION_SPEC.md` | provenance 공백이 존재함 | ODIAC2025 또는 다른 release 중 어느 것인지 | 원본 raster 또는 immutable archive 확보 전 release 확정 금지 |
| ODIAC2025 1 km GeoTIFF 값은 월별 `tonne carbon/cell` total이다 | E | ODIAC2025 README §2; official download page | 변수 단위·시간·공간 해상도 | 현재 legacy NPZ 값이 이 단위를 보존했다는 사실 | source header와 aggregation audit |
| ODIAC2025는 fossil-fuel combustion, cement production, gas flaring CO2를 포함한다 | E | ODIAC2025 official download page/README | target inventory의 포함 범위 | 모든 anthropogenic GHG, biogenic flux 또는 consumption-based emissions | paper target scope를 이 세 범주로 제한 |
| ODIAC는 직접 위성 CO2 측정값이 아니라 inventory-based spatial proxy이다 | D1 | Oda & Maksyutov (2011); Oda et al. (2018) | 국가 inventory를 point source·NTL로 공간 분해하는 방법 | local ground truth 또는 직접 flux measurement 정확도 | paper에서 `ODIAC-referenced inventory target` 사용 |
| legacy label은 `all_touched=True`로 선택된 ODIAC pixels의 산술평균이다 | P | recovered `src/preprocessing/raster_label.py`; `IMPLEMENTATION_SPEC.md` §3.5 | 실제 구현식 | source/target 정렬 시 결과가 무엇을 의미하는지 | synthetic raster regression test와 source metadata 확인 |
| `tC/source-cell/month`를 다른 polygon total로 옮길 때 산술평균은 일반적인 질량보존 연산이 아니다 | M | ODIAC2025 unit definition; mass-conservation identity in `REFERENCE_AUDIT.md` §4.4 | intersection-area weighting 필요성 | 현재 grid가 특수한 1:1 alignment라서 우연히 동일할 가능성의 배제 | area-weighted v2와 regional conservation test |
| clipped boundary cell의 target 정의가 없다 | U | recovered `grid.py`는 bbox intersection; source raster 부재 | full-square와 clipped-area 해석이 구분되지 않음 | 어느 정의가 논문 목적에 맞는지 | boundary policy 사전 결정 |
| `nightlight_mean`은 ODIAC construction proxy와 중복된다 | D1 | Oda & Maksyutov (2011); Oda et al. (2018); project feature config | ODIAC가 nighttime lights로 non-point emission을 배분함 | 프로젝트 VIIRS asset이 ODIAC2025 내부 asset과 정확히 동일함 | NTL 제거 ablation; exact ODIAC NTL provenance 문의/확인 |
| 발전소 개수·화석 capacity는 ODIAC point-source allocation과 개념적으로 중복된다 | D1 | Oda & Maksyutov (2011); recovered `vector_features.py` | 양쪽 모두 power-plant location/size 정보와 연관 | WRI GPPD가 ODIAC2025에 그대로 사용됨 | power features 제거 ablation; ODIAC point-source DB 확인 |
| ODIAC2025와 프로젝트가 같은 NTL/발전소 원자료를 썼는지는 미확정이다 | U | ODIAC2025 README는 proxy 종류만 명시; 내부 version 미기재 | exact dataset identity 공백 | 직접 변수 leakage의 정도 | provider documentation 또는 production manifest 확보 |
| NO2/SO2/CO는 ODIAC 공식 construction input으로 열거되지 않는다 | E | ODIAC2025 README; Oda et al. (2018) | 공식 공개 방법에 이 가스들이 allocation input으로 없음 | 내부 production pipeline 어디에도 쓰이지 않았다는 완전한 부정 | paper에서 `not documented as construction inputs`로 한정 |
| NO2/SO2/CO column은 CO2 emission의 직접 측정값이 아니다 | D2 | Yang et al. (2023); Zhang et al. (2024); GEE official product docs | co-emission proxy와 chemistry/transport 영향 | 이 프로젝트에서 predictive value가 없다는 주장 | ablation, meteorology limitation, spatial/temporal lag sensitivity |
| legacy GEE 결측치는 split 전에 월별 전체 cell 평균 또는 0으로 채워졌다 | P | recovered `cleaning.py`와 config | upstream preprocessing 순서 | 실제 결측 비율·성능 영향 | raw pre-imputation features 재구축·누수 영향 비교 |
| Phase 2 train-only preprocessing만으로 legacy upstream imputation leakage를 제거할 수 없다 | P | recovered `cleaning.py`; `src/cell_msca/data.py` | 이미 채워진 NPZ에 후단 train-only stats를 적용함 | 누수 효과 크기 | v2 raw feature pipeline 생성 |
| Bilotta et al. (2025)은 ODIAC를 target으로 한 직접 urban ML 선행연구다 | D1 | Bilotta et al. (2025) full text | ODIAC 기반 urban grid estimation, XGBoost/GCN, open data | Delhi monthly 1 km same-cell task와 동일함 | related-work table과 baseline comparison 포함 |
| Bilotta et al.의 70/15/15 split은 spatial block으로 확인되지 않는다 | U | Bilotta et al. (2025) full text | 비율은 제시되나 spatial assignment 세부 부재 | random split이었다는 단정 | 저자 code/supplement 확인 또는 `not reported`로 기술 |
| Gaughan et al. (2019)의 RF는 ODIAC target predictor가 아니다 | D2 | Gaughan et al. (2019) full text | RF가 WorldPop weighting surface 생성에 사용됨 | population과 ODIAC proxy 비교 자체가 무의미함 | Bilotta table의 해석을 그대로 재인용하지 않기 |
| Yang et al. (2023)의 R2=0.66은 ODIAC supervised prediction score가 아니다 | D2 | Yang et al. (2023) full text | space-based CO2–NO2 scene 관계와 mass balance | NO2가 urban CO2 proxy로 유용할 가능성의 부정 | target·metric을 원문대로 구분하여 인용 |
| Yang et al. (2019)은 GOSAT dXCO2와 GRNN으로 emission을 추정하고 ODIAC로 검증했다 | D1 | Yang et al. (2019) full text | 실제 satellite-to-ODIAC 계보 | 1 km monthly urban generalization | direct prior comparison 포함 |
| Mustafa et al. (2021)은 OCO-2 XCO2와 NPP로 ODIAC-referenced regional emissions를 추정했다 | D1 | Mustafa et al. (2021) full text | ODIAC target/reference ML precedent | pollution/socio-infrastructure token architecture | direct prior comparison 포함 |
| Zhang et al. (2022)은 ODIAC를 학습 target으로 한 global stacked RF를 제시했다 | D1 | Zhang et al. (2022) full text | 2014–2018 train, 2019 temporal test, NTL/XCO2 등 | Delhi 1 km monthly task 성능 | NTL circularity와 temporal split 비교 포함 |
| Zhang et al. (2024)의 TROPOMI NO2 모델은 EDGAR를 학습 target으로 쓴다 | D2 | Zhang et al. (2024) abstract/section snippets | NO2 기반 fine-grid CO2 estimation precedent | ODIAC supervised target precedent | paper에서 EDGAR/ODIAC를 혼동하지 않기 |
| Ou et al. (2026)은 ODIAC/EDGAR-driven monthly STRF와 NO2/CO/NTL를 사용한다 | D1 | Ou et al. (2026) full text | 매우 가까운 monthly multi-source inventory-target precedent | 1 km Delhi same-cell token/cross-attention이 이미 검증됨 | task·split·feature·metric 직접 비교 |
| 현재 문헌검색은 “first” 주장을 지원하지 않는다 | U | 본 감사 search log; 다수 D1 발견 | first claim의 근거 부족 | 모든 관련 문헌을 exhaustively 찾았다는 사실 | 별도 systematic prior-art search 전 first 금지 |
| ODIAC와 local city inventory는 도시·cell 수준에서 큰 차이를 보일 수 있다 | D2 | Gurney et al. (2019); Chen et al. (2020); Ahn et al. (2023) | ODIAC external validity 제한 | Delhi에서 동일한 편차 크기 | Delhi local inventory/top-down external validation |
| FT-Transformer는 연속형 scalar를 feature-specific token으로 직접 표현한다 | M | Gorishniy et al. (2021), numeric tokenizer equation | 7개 continuous feature를 별도 token으로 두는 선례 | 두 domain stream 구성·방향성 fusion | tokenizer unit tests와 FT-style baseline 고려 |
| TabTransformer의 Transformer는 주로 categorical embeddings를 문맥화한다 | M | Huang et al. (2020) full text | continuous features는 후단 MLP concat | continuous token cross-attention의 직접 선례 | paper의 architectural attribution 수정 |
| 3/4 feature grouping은 선행 모델이 정한 구조가 아니다 | H | FT-Transformer·TabTransformer에 해당 domain grouping 없음 | project-specific adaptation임 | grouping이 성능·과학 의미를 개선함 | all-feature/no-group 또는 alternative grouping ablation 검토 |
| 실제 project Stream A는 3 pollution features, Stream B는 4 socio-infrastructure features다 | P | split feature metadata; `src/cell_msca/data.py`; `IMPLEMENTATION_SPEC.md` | 정확한 feature order·stream membership | 이 의미 명칭이 연구적으로 최선임 | Phase 4 전에 모든 문서 명칭 통일 |
| MulT는 directional cross-modal attention의 일반 메커니즘을 지원한다 | M | Tsai et al. (2019) full text | Q stream이 K/V stream을 가중 집계하는 수학 | 이 도메인의 특정 방향 우월성 | MulT를 general mechanism 근거로만 인용 |
| `Q=infrastructure`, `K/V=environment`는 infrastructure token별 environment summary를 만든다 | M | scaled dot-product cross-attention equation; Tsai et al. (2019) | output length와 정보 흐름의 수학적 의미 | causal emission attribution | architecture docs에 수학적 표현 사용 |
| 특정 `infrastructure <- environment` 방향이 더 좋다는 주장은 미검증이다 | H | 직접 domain evidence 없음 | hypothesis status | 성능 또는 물리적 정당성 | concat/no-CA/reverse/bidirectional ablation |
| attention weight는 causal attribution 또는 source apportionment 증거가 아니다 | H | project에 causal identification design 부재 | 현재 해석의 근거 부족 | attention visualization의 진단적 유용성 | causal 표현 금지; 별도 XAI sanity checks |
| cell-fixed split은 같은 cell의 모든 월을 한 split에 둔다 | P | `splits/cell_fixed_seed42.csv`; v2 metadata; `src/cell_msca/splits.py` | same-cell repeated-measure leakage 방지 | 인접 cell의 공간 독립성 | assignment invariant 유지 |
| cell-fixed split은 공간 독립성을 보장하지 않는다 | M | Roberts et al. (2017) | spatially autocorrelated neighbors가 random cell folds를 넘을 수 있음 | 현재 성능 낙관성의 정확한 크기 | buffered/block robustness evaluation |
| block/buffer 거리는 empirical spatial autocorrelation을 바탕으로 정해야 한다 | M | Roberts et al. (2017); Valavi et al. (2019) | predictor/sample autocorrelation range 기반 선택 | 단 하나의 보편적 block size | variogram, anisotropy, origin/size sensitivity |
| 현재 buffered split은 block 24, buffer 16의 개발 설정이다 | P | `buffered_block_seed42.metadata.v2.json` | parameter, pair distance, dropped ratio, research status | 연구적으로 최종 적합함 | data-driven parameter로 새 immutable split 제안 |
| 현재 buffered split은 validation-test를 공간 분리하지 않는다 | P | metadata: minimum validation-test Chebyshev distance=1 | 두 held-out subset의 인접 가능성 | train과 held-out이 인접함 | split topology 재설계·지도 검토 |
| 현재 split protocol은 forecasting이 아니다 | P | cell-fixed assignment는 모든 연도의 월을 cell별로 나눔 | future time holdout 부재 | 시간 일반화가 낮다는 단정 | forecasting 주장 시 별도 chronological holdout |
| raw-target regression은 원 단위 risk를 직접 최적화하는 baseline이다 | M | loss definitions; LightGBM official docs | L1/MAE와 L2/MSE target-space objective | 어떤 raw loss가 최상인지 | L1/L2 선택을 estimand와 정렬 |
| log transformation의 적합성은 skew뿐 아니라 heteroscedasticity와 estimand에 달려 있다 | M | Manning & Mullahy (2001) | retransformation·mean specification 문제 | log1p가 이 데이터에서 우수함 | train-only distribution/residual diagnostic |
| `log1p(y/c)`의 scale `c`는 target unit와 함께 정의되어야 한다 | M | dimensional consistency; `src/cell_msca/target.py` | dimensionless log argument 필요 | 현재 `c=1`의 물리적 단위 | target unit 확정 후 scale 기록 |
| `1<p<2` Tweedie는 zero와 positive continuous outcome의 compound Poisson-gamma 형태를 허용한다 | M | Jorgensen (1987); LightGBM official docs | variance power family와 log link | 실제 ODIAC target이 이 분포를 따름 | zero mass·mean-variance diagnostic 및 p tuning |
| Huber loss는 작은 잔차에 quadratic, 큰 잔차에 linear penalty를 준다 | M | Huber (1964) | robust loss의 일반 근거 | log-Huber와 delta=1의 domain 적합성 | validation-only delta tuning; transformed-unit 기록 |
| Duan smearing은 log model의 원 단위 mean retransformation을 위한 비모수 방법이다 | M | Duan (1983) | smearing factor의 목적 | global factor가 이분산 자료에서도 조건부 unbiased임 | train residual conditional diagnostics; grouped sensitivity |
| Duan factor를 train residual만으로 계산하면 validation/test target leakage를 막는다 | M | validation methodology; current Phase 3 contract | selection data 분리 원칙 | factor가 통계적으로 최적임 | train-only invariant test 유지 |
| median/Duan 선택을 validation에서 하고 test 전에 동결하는 것은 leakage-free model selection이다 | M | standard validation separation; project frozen-selection design | test 비접근 선택 절차 | MAE 선택이 mean estimand와 항상 정렬됨 | primary estimand·selection metric 사전 선언 |
| ODIAC 원 단위가 연구 estimand이면 headline MAE/RMSE/R2/Bias는 inverse 후 그 단위에서 계산해야 한다 | M | estimand/reporting consistency; metric definitions | 원 단위 오차 해석 | 모든 연구에서 log metric이 부적절함 | original-unit primary와 log secondary 유지 |
| Spearman은 monotonic rank agreement를 측정하는 secondary metric이다 | M | Spearman (1904) | 순위 보존 평가 | magnitude error, bias, calibration, physical accuracy | headline에서 제외하고 secondary 유지 |
| 실제 target의 zero 비율·이분산·Tweedie 적합성은 현재 checkout에서 평가할 수 없다 | U | `data/v1_legacy` 부재 | 필수 진단 미실행 상태 | 특정 loss의 부적합 | 원본 data 확보 후 train-only EDA |
| `v1_legacy`의 물리적 원 단위 명칭은 확정되지 않았다 | U | split metadata target unit; source TIFF 부재 | 보수적 unit 문자열 필요 | tC/cell/month 또는 density 해석 | v2 provenance 해결 전 legacy unit 주장 금지 |
