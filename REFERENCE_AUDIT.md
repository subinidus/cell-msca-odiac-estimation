# Phase 3.5 Reference and Research-Claim Audit

감사일: 2026-08-23
대상 프로젝트: `msca-carbon-main`
범위: Cell-MSCA 구현 전 연구설계·인용·프로젝트 계보 검증만 수행
코드·테스트·설정·split·checkpoint·기존 결과 변경: 없음
학습 및 test split 평가: 수행하지 않음

## 1. 판정 요약

1. 현재 프로젝트가 **사용하려고 한 원자료**는 복구된 설정의 파일명 패턴상 `ODIAC2025` 1 km 월별 GeoTIFF이다(P). 그러나 프로젝트에 원본 GeoTIFF, 헤더, 체크섬, 다운로드 기록이 없으므로 **현재 NPZ가 실제로 어느 ODIAC release에서 생성되었는지는 확정할 수 없다(U)**.
2. ODIAC2025 1 km GeoTIFF의 공식 단위는 `tonne carbon/cell (monthly total)`이다(E). 따라서 `all_touched=True`로 선택된 원본 픽셀 값의 산술평균을 `1 km target-cell monthly total`로 해석하는 것은 일반적으로 질량보존 집계가 아니다(M). 현재 라벨은 논문에서 **`v1_legacy ODIAC-derived zonal mean`**으로만 기술해야 한다.
3. ODIAC는 직접 CO2 측정치가 아니다. 국가·연료별 인벤토리 총량을 발전소 위치, 야간조명 등으로 공간 분해한 **inventory-based, spatially disaggregated proxy target**이다(D1).
4. 프로젝트의 `nightlight_mean`은 ODIAC의 공간분해 proxy와 직접적인 개념 중복이 있고, 발전소 개수·화석발전 용량도 ODIAC point-source allocation과 개념적으로 겹친다(D1). 이 변수들을 이용한 높은 적합도는 독립 관측으로 재발견한 성능이 아니라 **ODIAC 공간배분 규칙을 부분적으로 재구성한 성능일 수 있다**. 정확히 같은 원자료를 공유하는지는 ODIAC2025 내부 point-source/NTL 버전이 공개 README만으로 확인되지 않아 U이다.
5. ODIAC 또는 다른 인벤토리를 학습·검증 대상으로 하는 위성·ML 연구는 이미 다수 존재한다. Bilotta et al. (2025), Yang et al. (2019), Mustafa et al. (2021), Zhang et al. (2022), Ou et al. (2026)이 직접적인 비교 대상이다(D1). 따라서 현재 검색으로 “first” 주장은 지원되지 않는다.
6. FT-Transformer는 연속형 수치 특성을 `b_j + x_j W_j` 형태의 개별 토큰으로 직접 지원한다(M). TabTransformer의 Transformer는 범주형 임베딩을 문맥화하고 연속형 특성은 후단 MLP에 연결하므로, 연속형 7개를 attention token으로 쓰는 직접 근거는 아니다(M).
7. MulT는 방향성 cross-modal attention의 일반 메커니즘을 지원한다(M). `Q=사회·인프라`, `K/V=대기오염`이라는 특정 방향의 도메인 우월성은 지원하지 않는다(H). 양 방향과 bidirectional ablation이 필요하다.
8. cell-fixed split은 동일 셀의 월별 중복이 split을 넘는 누수를 막는다(P). 그러나 인접 셀 간 공간 자기상관을 제거하지는 않는다(M). 현재 `block_size=24`, `buffer_cells=16` split은 연구적으로 동결되지 않았고 validation-test 최단 Chebyshev 거리가 1이며 전체 셀의 51.56%를 버린다(P). 개발용 robustness 후보이지 최종 독립성 근거가 아니다.
9. raw, log1p, Tweedie, Huber, Duan smearing은 모두 조건부로 정당화할 수 있다(M). 어느 방법이 적절한지는 target의 0 비율, 평균-분산 관계, 이분산성, 논문의 estimand가 평균인지 중앙값인지에 달려 있다. 현재 실제 NPZ가 checkout에 없어서 이 적합성은 검증되지 않았다(U).
10. 논문의 headline이 ODIAC 원 단위 오차라면 MAE/RMSE/R2/Bias는 inverse 후 원 단위에서 계산해야 한다. log-space 지표와 Spearman은 서로 다른 질문을 답하므로 secondary로 유지하는 설계가 타당하다(M). 현재 코드가 이를 구현한다는 사실은 P이다.

## 2. 근거 등급과 감사 방법

| 등급 | 의미 |
|---|---|
| D1 | target과 task가 실질적으로 같은 직접 선행연구 |
| D2 | 도메인 인접 선행연구 |
| M | 일반 방법론 근거 |
| E | 공식 소프트웨어·데이터·공학 문서 |
| P | 확인된 프로젝트 파일 근거 |
| H | 검증되지 않은 프로젝트 가설 |
| U | 현재 자료로 미해결 |

문헌 근거와 프로젝트 근거는 아래에서 분리한다. 문헌은 publisher/학회/공식 데이터 제공처의 원문을 우선했고, 접근 가능한 경우 본문 전체와 방법·한계·표를 확인했다. 프로젝트 근거는 파일 내용을 직접 읽은 결과만 사용했다. `references.bib`의 DOI와 URL은 2026-08-23에 DOI resolver, publisher, 학회, 공식 문서 페이지 또는 공개 원문 저장소에서 대조했다.

이번 검색은 ODIAC 공식 사이트, Crossref/DOI 메타데이터, publisher full text, PubMed/PMC, ACL Anthology, NeurIPS Proceedings, Copernicus, Wiley, Taylor & Francis, Elsevier와 논문 reference chain을 포함한다. 이는 체계적 문헌고찰 프로토콜로 등록된 exhaustive search가 아니므로 “최초”를 입증하지 않는다.

## 3. 프로젝트 파일 근거(P)와 감사 경계

### 3.1 직접 읽은 프로젝트 자료

- `IMPLEMENTATION_SPEC.md`, `PHASE1_1_REPORT.md`, `PHASE2_REPORT.md`, `PHASE2_1_REPORT.md`, `PHASE3_REPORT.md`
- `src/cell_msca/__init__.py`, `target.py`, `metrics.py`, `evaluate.py`, `data.py`, `splits.py`, `baselines.py`, `neural_baselines.py`, `experiment.py`
- `configs/baseline_example.json`
- `splits/cell_fixed_seed42.csv`, `splits/cell_fixed_seed42.metadata.v2.json`
- `splits/buffered_block_seed42.csv`, `splits/buffered_block_seed42.metadata.v2.json`
- 복구 전처리 archive `C:\Users\SAMSUNG\Downloads\carbon-hotspot-detection-main.zip`
  - SHA-256: `50CBFA18792D7813182A6F1B5E57B8D5B17BE4F736472DCED989BED13C9F7E42`
  - 내부 `configs/delhi_2022_2024.yaml`, `grid.py`, `gee_extract.py`, `cleaning.py`, `vector_features.py`, `raster_label.py`, `tensor_builder.py`

### 3.2 확인된 데이터·split 계약

| 항목 | 확인 내용 | 등급 |
|---|---|---|
| 데이터 버전 | `v1_legacy`, 2022-01~2024-12의 36개월 manifest | P |
| 공간 배열 | 월별 `[136,130]`, 유효 cell 17,157, 총 표본 617,652로 metadata에 기록 | P |
| Stream A | `no2_mean`, `so2_mean`, `co_mean` | P |
| Stream B | `nightlight_mean`, `urban_fraction`, `power_plant_count`, `fossil_capacity_mw` | P |
| target metadata | `ODIAC native value; exact target-cell interpretation unverified` | P |
| 데이터 hash | `c3a1889fd863117c3faa9125094267974dbacfe6fd4ca6c157dead2d7c874b62` | P |
| cell-fixed split | seed 42, train/validation/test cell 12,009/2,574/2,574, 실제 비율 약 70/15/15 | P |
| cell-fixed hash | `8b795ab11ad741a5e161b08ee1c2d44cefa1b21e5b6a0a724a7ca88a85cacb40` | P |
| buffered split | block 24, buffer 16; train/validation/test/dropped 3,984/2,458/1,868/8,847 | P |
| buffered 거리 | Chebyshev train-validation 17, train-test 17, validation-test 1 | P |
| buffered 상태 | `development_only_not_research_frozen` | P |

### 3.3 복구된 전처리 계보

1. `grid.py`는 EPSG:32643에서 1,000 m 사각 격자를 만들고 Delhi NCR bounding rectangle과 교차시킨다. 경계 셀은 잘린 polygon이 될 수 있다(P).
2. `gee_extract.py`는 각 기간의 image collection 평균을 만든 뒤 `reduceRegions(..., reducer=mean)`으로 셀별 평균을 구한다(P).
3. `cleaning.py`는 split 생성 전에 월별 전체 grid의 결측을 월 평균·중앙값 또는 0으로 채운다(P). 따라서 Phase 2의 train-only preprocessing이 있더라도 이미 채워진 legacy NPZ에서는 validation/test가 upstream 월 평균에 기여했을 가능성이 있다. 이는 `v1_legacy`의 전처리 누수 제한사항이다(P).
4. `vector_features.py`는 WRI GPPD를 이용해 모든 발전소 개수와 Coal/Gas/Oil/Petcoke 용량 합을 만든다(P).
5. `raster_label.py`는 ODIAC GeoTIFF에 대해 `zonal_stats(stats=[mean,max,min], all_touched=True)`를 실행하고 `mean`을 `label_reg`로 저장한다(P).
6. `tensor_builder.py`는 raw `label_reg`를 NPZ에 저장한다. log1p 값은 NPZ target으로 저장하지 않는다(P).

### 3.4 현재 감사로 확인할 수 없는 항목

프로젝트 checkout에 `data/` 또는 `data/v1_legacy/`가 없고 Downloads에서도 설정 패턴과 일치하는 ODIAC TIFF를 찾지 못했다. 따라서 다음은 U이다.

- 실제 사용된 ODIAC release와 정확한 36개 원본 파일
- 각 TIFF의 SHA-256/MD5, CRS, affine transform, pixel bounds, nodata, dtype
- 원본 ODIAC raster와 EPSG:32643 target grid의 정렬 관계
- 복구 코드가 현재 36개 NPZ를 실제 생성한 코드인지 여부
- NPZ 생성 시점의 GEE asset revision과 품질 필터
- 원본 결측치와 split 전 imputation의 실제 영향량

## 4. A. ODIAC target validity

### 4.1 공식 ODIAC2025 정의(E)

ODIAC2025 공식 배포 페이지와 README에 따르면 다음과 같다.

- 버전명: **ODIAC2025**
- release date: **2026-06-05**
- temporal coverage: **2000-01~2024-12**
- 변수: fossil fuel combustion, cement production, gas flaring에서 발생하는 CO2 emissions
- temporal resolution: **monthly**
- 공간 해상도·형식: **1 km GeoTIFF**, **1 degree netCDF**
- 1 km GeoTIFF 단위: **tonne carbon/cell, monthly total**
- 1 km GeoTIFF 범위: land emission만 포함, international aviation·marine bunker 제외
- dataset DOI: **10.17595/20170411.001**

공식 자료: [ODIAC2025 배포 페이지](https://db.cger.nies.go.jp/dataset/ODIAC/DL_odiac2025.html), [ODIAC2025 README](https://db.cger.nies.go.jp/dataset/ODIAC/readme/readme_ODIAC2025_20260605.txt), [data policy와 공식 citation](https://db.cger.nies.go.jp/dataset/ODIAC/data_policy.html).

복구 설정이 `odiac2025_1km_excl_intl_{yy}{mm}.tif` 패턴을 가진다는 것은 P이다. 이 패턴이 공식 파일명과 일치한다는 것은 E이다. 다만 원본 파일 자체가 없으므로 “현재 NPZ는 ODIAC2025로 생성되었다”가 아니라 “복구 설정은 ODIAC2025를 지시한다”까지만 안전하다.

### 4.2 ODIAC 구축 방식(D1)

Oda and Maksyutov (2011), Oda et al. (2018), ODIAC2025 README를 함께 보면 구축 개념은 다음과 같다.

1. 국가·연료별 fossil-fuel CO2 총량을 인벤토리 통계로 산정한다.
2. 대형 point source는 발전소 등 지리 위치 자료로 배치한다.
3. non-point source는 satellite nighttime lights 등의 공간 proxy로 분배한다.
4. 월별 계절성은 CDIAC monthly 자료와 2020년 이후 Carbon Monitor 보완 자료를 이용한다.
5. 최신 연도는 Energy Institute 통계로 국가 총량을 projection한다.

따라서 ODIAC pixel은 위성에서 직접 측정한 CO2 concentration이나 flux가 아니다. 다음 세 표현을 구분해야 한다.

| 표현 | 이 프로젝트 target에 해당하는가 | 설명 |
|---|---:|---|
| direct emissions measurement | 아니오 | stack sensor, chamber, flux tower 또는 atmospheric inversion의 직접 관측 결과가 아님 |
| inventory estimate | 예 | 국가 통계·배출계수·부문 자료를 기반으로 총량을 추정 |
| spatially disaggregated proxy target | 예 | point-source 위치와 야간조명 등 proxy로 총량을 1 km cell에 분배 |

논문에서는 `ODIAC-referenced fossil-fuel CO2 inventory target` 또는 `ODIAC-derived target`이 안전하다. `satellite CO2 ground truth`, `measured CO2`, `true emissions`는 제거해야 한다.

### 4.3 프로젝트 입력과 ODIAC 구축정보 중복

| 프로젝트 입력 | 공식 변수 정의/프로젝트 산출 | ODIAC 구축정보와의 관계 | 판정 |
|---|---|---|---|
| `no2_mean` | Sentinel-5P OFFL tropospheric NO2 column monthly mean | ODIAC 구축 입력으로 공식 문서에 열거되지 않음. 연소 동시배출 proxy이나 수송·화학·비화석원 영향 존재 | D2/E; 직접 중복 증거 없음 |
| `so2_mean` | Sentinel-5P OFFL SO2 column monthly mean | ODIAC 구축 입력으로 열거되지 않음. 연료 황 함량·산업·화산 등 복수 원인 | D2/E; 직접 중복 증거 없음 |
| `co_mean` | Sentinel-5P OFFL CO column monthly mean | ODIAC 구축 입력으로 열거되지 않음. 불완전연소 proxy이나 수송·산불 등 영향 | D2/E; 직접 중복 증거 없음 |
| `nightlight_mean` | VIIRS monthly `avg_rad` cell mean | ODIAC non-point spatial allocation에 nighttime lights가 직접 사용됨 | D1/P; 강한 construction-proxy overlap |
| `urban_fraction` | MODIS MCD12Q1 class 13 fraction | 공식 ODIAC 구축 입력으로 열거되지 않음. NTL·인구·활동과 상관 가능 | E/P; 직접 중복 미확인 |
| `power_plant_count` | WRI GPPD 내 cell 발전소 수 | ODIAC는 point-source locations를 사용 | D1/P; 개념 중복. 동일 database인지는 U |
| `fossil_capacity_mw` | WRI GPPD Coal/Gas/Oil/Petcoke capacity 합 | ODIAC 발전소 배출량 배치와 강하게 관련 | D1/P; 개념 중복. 동일 capacity/emission 자료인지는 U |

공식 feature 문서는 [NO2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2), [SO2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_SO2), [CO](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_CO), [VIIRS](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG), [MCD12Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1), [WRI GPPD](https://developers.google.com/earth-engine/datasets/catalog/WRI_GPPD_power_plants)이다.

필수 결과 해석은 “독립적인 물리적 CO2 측정 예측”이 아니라 “ODIAC inventory field를 공개 proxy로 근사”이다. NTL·발전소 제거 실험은 overlap의 영향을 정량화하지만, 나머지 변수도 도시활동과 상관하므로 완전한 독립 검증은 아니다.

### 4.4 현재 라벨 집계식의 물리적 타당성

복구 코드가 저장하는 값은 다음 산술평균이다.

\[
y_{legacy}(g,t)=\frac{1}{|P(g)|}\sum_{p\in P(g)}E(p,t),
\]

여기서 `P(g)`는 `all_touched=True`로 target polygon `g`와 닿는 모든 source pixel이고, 공식 ODIAC2025 GeoTIFF의 `E(p,t)`는 `tC/source-cell/month`이다.

source가 cell total일 때 target polygon의 질량보존 total은 일반적으로 다음처럼 면적 교차비를 사용해야 한다.

\[
y_{total}(g,t)=\sum_p E(p,t)\frac{A(p\cap g)}{A(p)}.
\]

따라서 산술평균은 다음 특수조건이 실제 raster metadata로 확인될 때만 source cell total과 우연히 같을 수 있다: target cell이 하나의 source pixel과 정확히 일치하고, reprojection·경계 clipping·복수 touched pixel이 없을 때. 현재 grid는 EPSG:32643에서 만들고 bounding rectangle으로 잘랐으며 source raster 정렬을 확인할 수 없으므로 이 조건은 확인되지 않는다(U). `all_touched`는 교차면적 가중을 수행하지 않는다.

`v2_corrected`에는 다음 검증이 필요하다.

- source raster CRS·transform·pixel area 기록
- target cell을 full 1 km square로 정의할지 clipped study-boundary area로 정의할지 명시
- intersection-area weighting을 통한 monthly tC 보존
- 전체 covered region에서 source mass와 target mass의 conservation identity 검증
- tC를 tCO2로 보고할 경우 `44/12` 변환과 단위명 명시
- 경계·nodata·해안·excluded bunker 처리 규칙 고정

## 5. B. 직접 및 인접 선행연구

### 5.1 직접 비교(D1)

| 연구 | task·target | 입력 | 범위·split | 모델·지표 | 이 프로젝트와 다른 점/한계 |
|---|---|---|---|---|---|
| Bilotta, Ipsaro Palesi & Nesi (2025), *Expert Systems with Applications* 291:128598, DOI [10.1016/j.eswa.2025.128598](https://doi.org/10.1016/j.eswa.2025.128598) | ODIAC 기반 연간 도시 grid CO2 추정 | OpenStreetMap·도시서비스·사회경제, GPS variant | Bologna·Florence, 2019, 약 700 m; 70/15/15 관측 분할이나 공간 block 여부 미기재 | OLS/LR, XGBoost, GCN; MAE/RMSE/MAPE/R2. 최상 variant MAPE 약 8%, R2 약 0.96 | 월별 단일-cell 대기 trace gas task가 아님. ODIAC를 satellite true value로 부르는 표현은 ODIAC 공식 정의와 불일치. split 공간독립성 U |
| Yang et al. (2019), *Sensors* 19:1118, DOI [10.3390/s19051118](https://doi.org/10.3390/s19051118) | GOSAT dXCO2로 anthropogenic emission을 학습하고 2015 결과를 ODIAC2015a로 검증 | GOSAT XCO2 anomaly | China, 1 degree, annual; 2010–2014 train, 2015 holdout | GRNN; ODIAC와 상관·오차 비교 | 매우 거친 annual regional scale. local 1 km urban task가 아님 |
| Mustafa et al. (2021), *Atmospheric Measurement Techniques* 14:7277–7290, DOI [10.5194/amt-14-7277-2021](https://doi.org/10.5194/amt-14-7277-2021) | OCO-2와 NPP로 regional emission 추정, ODIAC 사용 | OCO-2 XCO2, MODIS NPP | East/West Asia, 2015–2019, regional grid | GRNN; ODIAC 일치도·오차 | 1 km 도시 monthly split이 아니며 대기 transport 처리 한계 |
| Zhang et al. (2022), *Remote Sensing* 14:3899, DOI [10.3390/rs14163899](https://doi.org/10.3390/rs14163899) | ODIAC 1 degree annual grid를 학습하여 2019 emission 추정 | dXCO2, VIIRS NTL, ecosystem flux, SIF, EVI | global; 2014–2018 train, 2019 temporal test | two-layer stacked RF; grid R2=0.766, RMSE=0.359; third-party grid R2=0.665 | NTL-ODIAC construction overlap을 논문도 인정. coarse annual global task |
| Ou et al. (2026), *International Journal of Digital Earth* 19(1):2639803, DOI [10.1080/17538947.2026.2639803](https://doi.org/10.1080/17538947.2026.2639803) | ODIAC/EDGAR inventory-driven monthly gridded emission 추정 | XCO2 anomaly, NTL, NDVI, wind, NO2, CO, space/time features | China 0.1 degree, 2019–2022; first 36 months train, final 12 months temporal holdout; spatiotemporal blocked tuning | STRF vs LR/GRNN/RF/XGBoost; ODIAC variant R2=0.943, MSE/MAE/Bias | 현재 프로젝트보다 거칠지만 monthly·multi-source·ODIAC target이 매우 근접. 좌표·이웃/시간 특성 사용. ODIAC NTL 중복 존재 |

Bilotta et al.은 가장 직접적인 urban open-data comparator이며 반드시 related work와 baseline discussion에 포함해야 한다. 그러나 현재 프로젝트가 단일 도시·1 km·월별·동일-cell 7개 scalar를 사용한다는 차이는 task definition의 차이이지 자동적인 novelty 증명이 아니다.

### 5.2 추가 직접(D1)·근접(D2) 연구

- **Tan et al. (2024)**, “Estimation of Carbon Emissions in Various Clustered Regions of China Based on OCO-2 Satellite XCO2 Data and Random Forest Modelling,” *Atmospheric Environment* 338:120860, DOI [10.1016/j.atmosenv.2024.120860](https://doi.org/10.1016/j.atmosenv.2024.120860). XCO2 anomaly와 ODIAC로 영역을 군집화하고 다원 remote-sensing feature로 cluster별 RF를 적합해 약 R2 0.6을 보고한다. full text에 접근하지 못해 target 생성·split 세부는 U이며 D2로 보수적으로 분류한다.
- **Zhang et al. (2024)**, “Estimating global 0.1 degree scale gridded anthropogenic CO2 emissions using TROPOMI NO2 and a data-driven method,” *Science of the Total Environment* 949:175177, DOI [10.1016/j.scitotenv.2024.175177](https://doi.org/10.1016/j.scitotenv.2024.175177). TROPOMI NO2, CO2/NOx ratio, LSTM과 RF를 사용하지만 RF 학습 target은 **EDGAR**, ODIAC가 아니다. NO2 기반 직접 인접연구(D2)이다.
- **Yang et al. (2023)**, “Using Space-Based CO2 and NO2 Observations to Estimate Urban CO2 Emissions,” *JGR Atmospheres* 128:e2022JD037736, DOI [10.1029/2022JD037736](https://doi.org/10.1029/2022JD037736). OCO-3 XCO2–TROPOMI NO2 관계, NDCF, mass-balance로 Buenos Aires·Melbourne·Mexico City 배출을 추정한다. ODIAC는 GEOS-CF prior/urban outline/reference에 들어갈 뿐 supervised target이 아니다(D2).
- **Gaughan et al. (2019)**, “Evaluating nighttime lights and population distribution as proxies for mapping anthropogenic CO2 emission in Vietnam, Cambodia and Laos,” *Environmental Research Communications* 1:091006, DOI [10.1088/2515-7620/ab3d91](https://doi.org/10.1088/2515-7620/ab3d91). ODIAC2017 non-point allocation과 population-based allocation을 비교한다. 논문의 Random Forest는 WorldPop population weighting surface 생성에 속하며 ODIAC를 예측하는 RF가 아니다(D2).
- **Hsu et al. (2022)**, “Predicting European cities' climate mitigation performance using machine learning,” *Nature Communications* 13:7487, DOI [10.1038/s41467-022-35108-5](https://doi.org/10.1038/s41467-022-35108-5). self-reported city inventory가 target이고 ODIAC는 predictor 중 하나다. XGBoost로 annual city emissions를 다루므로 target 방향이 반대인 D2이다.

### 5.3 Bilotta et al.의 prior-work 표를 1차 자료로 추적한 결과

| 2차 서술 | 1차 자료 확인 | 판정 |
|---|---|---|
| Gaughan et al.이 population으로 ODIAC를 RF 예측하고 MSE 0.78을 냈다 | 원문은 NTL 기반 ODIAC non-point map과 WorldPop 기반 배분을 비교한다. RF는 WorldPop weighting surface 제작 단계다 | 해당 표를 그대로 인용하면 안 됨 |
| Yang et al. (2023)의 target은 ODIAC이고 linear regression R2=0.66이다 | 원문은 관측 scene의 CO2-NO2 관계와 mass-balance를 다룬다. ODIAC supervised target score가 아니다 | 해당 R2를 ODIAC 예측 성능으로 인용하면 안 됨 |
| satellite NO2가 ODIAC를 직접 예측한 선행연구다 | Zhang et al. (2024)은 TROPOMI NO2를 쓰지만 target은 EDGAR. Ou et al. (2026)은 NO2/CO와 ODIAC variant를 함께 사용 | 논문별 target을 분리해 서술해야 함 |

실제 ODIAC-target ML 계보는 Yang et al. (2019: GOSAT-XCO2/GRNN), Mustafa et al. (2021: OCO-2+NPP/GRNN), Zhang et al. (2022: multi-source/stacked RF), Ou et al. (2026: monthly STRF)로 추적된다. 따라서 본 연구의 차별점은 이 연구들과 명시적으로 비교해 실험으로 입증해야 한다.

### 5.4 ODIAC와 local inventory 비교(D2)

| 연구 | 범위 | 핵심 결과와 이 프로젝트에 주는 제한 |
|---|---|---|
| Gurney et al. (2019), *JGR Atmospheres* 124:2823–2840, DOI [10.1029/2018JD028859](https://doi.org/10.1029/2018JD028859) | LA, Baltimore, Indianapolis, Salt Lake City; ODIAC2013a vs Hestia | 도시 총량 차이 -1.5~+20.8%, 1 km cell median difference 47~84%, 공간상관 0.34~0.68. point source와 road 배분 오차가 큼 |
| Chen et al. (2020), *Carbon Balance and Management* 15:9, DOI [10.1186/s13021-020-00146-3](https://doi.org/10.1186/s13021-020-00146-3) | Delhi 포함 14개 도시 | global downscaling은 도시별 sector·경계 차이로 local inventory와 편차가 크며 ODIAC는 원래 urban policy monitoring용으로 설계되지 않음 |
| Ahn et al. (2023), *Environmental Research Letters* 18:034032, DOI [10.1088/1748-9326/acbb91](https://doi.org/10.1088/1748-9326/acbb91) | 78 C40 도시; GPC vs ODIAC/EDGAR | ODIAC-GPC relative difference 평균 12%, 표준편차 62%. first-order estimate에는 쓸 수 있으나 정책 추세 추적의 독립 기준으로 제한 |

이 결과들은 ODIAC를 label로 잘 맞추는 것과 실제 도시 인벤토리 정확도를 분리해야 함을 보인다. ACK paper에 정책 효용을 언급하려면 local Delhi inventory 또는 독립 top-down 관측으로 외부검증해야 한다.

## 6. C. Feature-token models

### 6.1 FT-Transformer(M)

Gorishniy et al. (2021), “Revisiting Deep Learning Models for Tabular Data,” *NeurIPS 34*, pp. 18932–18943, [공식 원문](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)은 연속형 feature `x_j`를 다음처럼 feature-specific affine embedding으로 토큰화한다.

\[
T_j=b_j+x_jW_j.
\]

따라서 각 scalar를 별도 token으로 두는 것은 직접적인 수치-feature token precedent이다. 그러나 이 논문은 NO2/SO2/CO와 NTL/urban/power를 두 domain stream으로 나누거나 방향성 cross-attention을 선택하지 않는다. 그 부분은 H이다.

### 6.2 TabTransformer(M)

Huang et al. (2020), “TabTransformer: Tabular Data Modeling Using Contextual Embeddings,” arXiv:2012.06678, DOI [10.48550/arXiv.2012.06678](https://doi.org/10.48550/arXiv.2012.06678)은 Transformer로 **categorical embeddings**를 문맥화한다. continuous features는 contextual categorical embedding과 함께 후단 MLP에 concatenation된다. 따라서 “TabTransformer가 연속형 scalar token attention을 직접 지원한다”는 주장은 부정확하다.

### 6.3 프로젝트 적용의 정당한 범위

- 정당한 직접 precedent: 7개 연속형 scalar를 각각 feature-specific token으로 embedding하는 것(FT-Transformer).
- 일반적 동기: heterogeneous tabular features 사이 상호작용을 attention으로 학습하는 것(FT-Transformer/TabTransformer).
- 프로젝트 고유 adaptation: 3개 대기오염 feature와 4개 사회·인프라 feature를 두 stream으로 나누는 것(H).
- 프로젝트 고유 adaptation: stream 사이에만 cross-attention을 두는 것(H).

실제 프로젝트 파일에서 3-token Stream A는 `NO2/SO2/CO`이고, 4-token Stream B는 `NTL/urban/power plant/capacity`이다(P). 즉 현재 README·spec의 의미는 **3 pollution/environmental tokens, 4 socio-infrastructure tokens**이다. “3-token infrastructure, 4-token environmental”이라는 반대 표현 중 어느 것을 채택할지는 U이며 Phase 4 전에 하나로 통일해야 한다.

## 7. D. Directional cross-attention

Tsai et al. (2019), “Multimodal Transformer for Unaligned Multimodal Language Sequences,” *ACL 2019*, pp. 6558–6569, DOI [10.18653/v1/P19-1656](https://doi.org/10.18653/v1/P19-1656)은 directional pairwise cross-modal attention의 직접 방법론 precedent다(M).

인프라 token 행렬을 `X_I`, 환경/오염 token 행렬을 `X_E`라고 하면:

\[
Q_I=X_IW_Q,\quad K_E=X_EW_K,\quad V_E=X_EW_V,
\]

\[
Y_{I\leftarrow E}=\operatorname{softmax}\left(\frac{Q_IK_E^\top}{\sqrt{d_k}}\right)V_E.
\]

`Q=infrastructure`, `K/V=environment`의 수학적 의미는 다음과 같다.

- output 길이는 infrastructure token 수다.
- 각 infrastructure query가 모든 environmental key와의 유사도로 가중치를 만든다.
- 해당 가중치로 environmental value의 가중합을 받아 infrastructure representation을 갱신한다.

이는 인과적 “오염을 인프라에 귀속”을 자동으로 뜻하지 않는다. attention weight는 학습된 표현의 결합 계수이며 causal attribution 또는 source apportionment의 증거가 아니다. MulT는 일반 cross-modal 방향성을 지원하지만 이 도메인의 특정 `I <- E` 방향이 reverse보다 우수하다고 보이지 않는다. 방향 선택은 H이다.

필수 ablation은 다음과 같다.

1. concat MLP: attention이 없는 tabular nonlinear baseline
2. no-cross-attention: 동일 tokenizer·self/within-stream encoder·유사 parameter budget, 마지막에 concat
3. reverse: `E <- I`, 즉 Q=environment, K/V=infrastructure
4. bidirectional: `I <- E`와 `E <- I`를 모두 사용

parameter count, hidden dimension, optimizer, train budget, validation rule을 가능한 한 맞춰야 방향 효과와 capacity 효과를 구분할 수 있다. attention map은 보조 진단으로만 쓰고 causal 설명으로 제시하지 않는다.

## 8. E. Spatial validation

Roberts et al. (2017), “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” *Ecography* 40:913–929, DOI [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)은 의존 구조가 있는 자료에서 random CV가 예측오차를 낙관할 수 있으므로 예측 목적에 맞춘 blocking을 권고한다(M). Valavi et al. (2019), “blockCV,” *Methods in Ecology and Evolution* 10:225–232, DOI [10.1111/2041-210X.13107](https://doi.org/10.1111/2041-210X.13107)은 predictor 또는 sample의 경험적 공간 자기상관 범위를 block 크기 선택의 출발점으로 사용할 수 있음을 보인다(M).

### 8.1 cell-fixed split의 역할과 한계

현재 cell-fixed split은 동일 `(row,col)`의 36개월을 하나의 split에만 둔다(P). 이는 same-cell repeated-measure leakage를 막는다. 그러나 서로 인접한 cell은 서로 다른 split에 들어갈 수 있고, ODIAC·NTL·대기 trace gas·urban land cover가 공간적으로 자기상관되어 있으면 train과 validation/test가 매우 유사할 수 있다. 따라서 cell-fixed는 **cell identity independence**이지 **spatial independence**가 아니다.

### 8.2 buffer/block 거리 선택

legacy 16x16 patch size는 공간 자기상관의 경험적 범위가 아니므로 buffer 근거로 사용하면 안 된다. Phase 3.6에서 다음 절차를 사전 정의해야 한다.

1. split 전에 target과 각 predictor의 empirical variogram/correlogram을 km 단위로 추정한다.
2. robust estimator, anisotropy, 계절별 범위, nonstationarity를 점검한다.
3. 후보 block size와 buffer를 최대 relevant autocorrelation range를 중심으로 정한다.
4. 여러 block origin·fold assignment에서 cell 수, target 분포, covariate shift를 확인한다.
5. 모델 적합 후 train residual의 spatial autocorrelation도 진단하되, test 성능을 보고 parameter를 고르지 않는다.
6. 거리별 sensitivity 결과를 함께 보고한다.

### 8.3 현재 buffered split 평가

현재 metadata상 `block_size=24`, `buffer_cells=16`은 명시적으로 development-only다(P). train-validation과 train-test는 17-cell Chebyshev 거리지만 validation-test는 1이고, 51.56%의 cell을 dropped 처리한다. 판정은 다음과 같다.

- 구현 가능성 smoke/robustness 후보: 적합
- 연구적으로 확정된 spatial holdout: 부적합
- train-to-heldout 근접 누수 감소: 일부 지원
- validation/test가 서로 공간 독립이라는 증거: 없음
- 실제 spatial autocorrelation range에 대응한다는 증거: 없음
- 표본 손실과 domain coverage bias가 허용 가능하다는 증거: 없음

최종 모델 선택은 cell-fixed validation에서 하고 buffered split을 robustness evaluation으로 사용할 수는 있으나, 그러려면 buffer를 경험적으로 정하고 validation/test 역할·최소거리·dropped 영역의 공간분포를 다시 설계해야 한다.

### 8.4 forecasting 표현

현재 cell-fixed split에는 2022–2024의 월이 train/validation/test 모두에 존재한다(P). 미래 월을 통째로 보류하지 않으므로 forecasting이 아니다. `spatial interpolation/reconstruction of an ODIAC-referenced monthly field` 또는 `same-period cell-level estimation`으로 기술해야 한다. forecasting을 주장하려면 과거 월 train, 이후 연속 월 validation/test의 진정한 temporal holdout이 별도로 필요하다.

## 9. F. Losses, transformations, and evaluation

| 방법 | 방법론적 근거 | 적절한 조건 | 가정·주의점 | 현재 판정 |
|---|---|---|---|---|
| raw-target regression | 원 단위 risk를 직접 최적화. LightGBM은 L2/L1 objective를 공식 지원(E) | 원 단위 MAE/RMSE가 연구 목적일 때 | L1은 조건부 중앙값, L2는 조건부 평균을 겨냥함. 큰 값과 이분산 영향이 다름 | baseline으로 필요(M) |
| log1p regression | Manning & Mullahy (2001)는 skewed positive outcome에서 log model과 retransformation의 estimand·이분산 문제를 다룸(M) | nonnegative이고 scale 차이가 크며 상대오차 구조가 유용할 때 | `log1p(y/c)`의 `c`는 물리 단위가 있어야 함. 원 단위 평균은 단순 exp inverse와 다름 | target unit·`c=1` 의미 미해결(U) |
| Tweedie regression | Jorgensen (1987)의 exponential dispersion model; LightGBM은 log-link Tweedie와 `1<p<2`를 지원(M) | 정확한 0과 양의 연속값이 함께 있고 `Var(Y)=phi*mu^p`가 근사될 때 | 비음수 필요. 단순 skewness만으로 정당화되지 않음. zero mass와 mean-variance plot 필요 | 실제 target 분포 부재로 적합성 U |
| Huber loss | Huber (1964)의 quadratic-near-zero, linear-tail robust loss(M) | outlier 영향과 작은 잔차 민감도의 절충이 필요할 때 | delta는 transformed scale에서 의미가 달라짐. domain-specific 정당화는 아님 | `log_huber`는 후보, delta validation 필요 |
| Duan smearing | Duan (1983)의 비모수 retransformation(M) | log model에서 원 단위 conditional mean을 복원하려 할 때 | global factor는 residual distribution이 x에 따라 변하지 않는 근사가 필요. 이분산이면 conditional/grouped smearing 검토 | train residual만으로 factor 추정, validation으로 mode 선택 가능 |

### 9.1 raw와 log estimand

raw-L1은 원 단위 conditional median에 가깝고, raw-L2는 conditional mean에 가깝다. log-L1/log-Huber 후 median inverse는 transformed conditional center를 원 단위로 되돌린 값이다. Duan smearing은 log residual의 평균 지수값을 이용해 원 단위 mean retransformation bias를 보정한다. “log가 항상 skew에 더 좋다” 또는 “Duan이 항상 정확하다”는 주장은 근거가 없다.

### 9.2 median inverse와 Duan 선택

방법론은 하나의 보편적 선택 규칙을 강제하지 않는다. 먼저 논문 estimand를 정해야 한다.

- 원 단위 conditional median 또는 MAE 최소화가 1차 목적이면 median inverse가 자연스럽다.
- 원 단위 conditional mean/총량의 unbiased recovery가 목적이면 Duan이 더 관련 있으나 residual 구조가 적합해야 한다.
- 두 방식을 validation original-unit MAE로 비교하는 방식은 test leakage를 피하는 model-selection procedure로 허용된다(M). 현재 코드가 이 절차를 구현한다는 사실은 P이다. 다만 mean estimand를 선언하고 MAE만으로 Duan을 탈락시키면 estimand와 selection metric이 어긋날 수 있다.
- Duan factor는 checkpoint 확정 후 **train residual만** 사용해야 한다(M). validation은 median/Duan 선택에만 쓰고 test 전에 inverse mode와 factor를 동결한다. 현재 코드 계약은 P이다.

권고: ACK paper에서 primary estimand를 먼저 선언하고, headline MAE 기준 selection을 사용할지 mean recovery 기준을 사용할지 사전 등록한다. median과 Duan 결과를 모두 supplementary로 남긴다.

### 9.3 metric 단위

연구 질문이 ODIAC target의 원 단위 오차라면 prediction을 inverse한 뒤 같은 원 단위에서 MAE, RMSE, R2, Bias를 계산해야 한다. log-space MAE/RMSE/R2는 transformed outcome에 대한 별도 성능이며 원 단위 오차를 대체하지 않는다. 이는 지표 자체의 보편 규칙이 아니라 **선언한 estimand와 보고 단위를 일치시키는 원칙**이다.

현재 `src/cell_msca`는 original-unit MAE/RMSE/R2/Bias를 headline, log-space metrics와 Spearman을 secondary로 분리하고 저장 prediction에서 재계산한다(P). 이 구조는 유지할 수 있다. 단, `v1_legacy` unit가 아직 미확정이므로 headline 단위명은 `ODIAC-derived legacy value`로 보수적으로 적어야 한다.

### 9.4 Spearman의 역할

Spearman (1904)의 rank correlation은 monotonic rank agreement를 평가한다(M). 크기 보정, bias, 물리 단위 오차, calibration을 측정하지 않는다. hotspot ranking과 순위 안정성의 보조 정보로 유용하지만 headline emission accuracy가 될 수 없다. 현재 코드의 secondary metric 분류는 P이다.

## 10. 원문 접근 상태

### 10.1 본문 전체를 확인한 자료

- ODIAC2025 official release page, README, data policy
- Oda & Maksyutov (2011), Oda et al. (2018)
- Bilotta et al. (2025)
- Yang et al. (2019), Mustafa et al. (2021), Zhang et al. (2022), Ou et al. (2026)
- Gaughan et al. (2019), Yang et al. (2023)
- Gurney et al. (2019), Chen et al. (2020), Ahn et al. (2023), Hsu et al. (2022)
- Gorishniy et al. (2021), Huang et al. (2020), Tsai et al. (2019)
- Roberts et al. (2017), Valavi et al. (2019)
- Duan (1983), Huber (1964), Spearman (1904)
- LightGBM official parameter documentation
- Google Earth Engine official catalog pages for all seven feature sources

Manning & Mullahy (2001)은 journal metadata와 동일 원고의 공개 NBER working-paper 전문을 확인했다. Journal of Health Economics version의 publisher 본문은 metadata/abstract 범위만 접근 가능했다.

### 10.2 abstract 또는 metadata만 확인한 자료

- Tan et al. (2024): publisher abstract, bibliographic metadata, DOI만 확인. full text 미접근.
- Zhang et al. (2024, TROPOMI NO2): publisher abstract·section snippets·PubMed metadata를 확인했으나 subscription 본문 전체는 미접근.

이 두 논문에 대해서는 세부 split이나 retransformation처럼 abstract에 없는 내용을 주장하지 않았다.

## 11. 결론

### 11.1 ACK paper에 안전한 주장

1. 본 연구는 Delhi NCR의 동일 1 km cell에서 얻은 7개 공개 proxy feature로 월별 **ODIAC-referenced inventory target**을 추정하는 문제를 다룬다. `v1_legacy`에는 주변 셀·좌표·hotspot label을 모델 입력으로 사용하지 않는다(P).
2. ODIAC는 직접 관측이 아니라 국가 인벤토리를 point source와 NTL 등으로 공간 분해한 fossil-fuel CO2 proxy target이다(D1).
3. 각 연속형 scalar를 개별 feature token으로 embedding하는 설계는 FT-Transformer에서 방법론적 동기를 얻는다(M).
4. 두 stream 사이 방향성 attention은 MulT의 일반 cross-modal mechanism에서 동기를 얻는다(M). 특정 stream 구성과 방향의 우월성은 H이다.
5. train-mean, raw/log1p/Tweedie LightGBM, concat MLP 및 방향 ablation을 동일 provenance-locked split에서 비교한다는 실험설계는 설명 가능하다(P). 성능 우월성은 실험 전에는 주장할 수 없다.
6. cell-fixed split은 같은 cell의 월별 표본이 split을 넘지 않게 한다(P). 공간 독립성은 별도 robustness protocol로 평가해야 한다(M).
7. headline metric은 inverse 후 원 단위 MAE/RMSE/R2/Bias이고 Spearman과 log-space metric은 secondary라는 방법론적 원칙은 M이다. 현재 구현 상태는 P이다.

### 11.2 약화하거나 제거해야 할 주장

- “satellite CO2 measurement/ground truth/true emissions를 예측한다” → “ODIAC-referenced inventory proxy를 근사한다.”
- “본 연구가 최초의 …” → 제거. exhaustive documented search가 없고 직접 선행연구가 존재한다.
- “Q=인프라 방향이 물리적으로 올바르다/배출을 귀속한다” → 검증할 hypothesis로 약화.
- “attention weight가 원인 또는 정책 lever를 설명한다” → 제거. causal/source-apportionment 근거 없음.
- “cell-fixed split이 spatially independent하다” → 제거.
- “16-cell buffer가 과학적으로 정당화되었다” → 제거. empirical autocorrelation로 재선정.
- “2022–2024를 예측하므로 forecasting이다” → 제거. true temporal holdout 없음.
- “높은 R2가 실제 도시 배출 정확도를 증명한다” → 제거. ODIAC construction-proxy overlap과 local inventory mismatch 존재.
- README의 기존 16x16 성능·해석을 새 1x1 Cell-MSCA 논문 결과처럼 인용 → 제거. legacy comparison으로만 분리.

### 11.3 남아 있는 novelty 후보

아래는 검증 전 후보이며 novelty claim이 아니다(H).

- Delhi NCR 1 km monthly ODIAC reconstruction에서 같은 cell의 7개 scalar만 사용하는 feature-token dual-stream 설계
- 3 pollution/environment tokens와 4 socio-infrastructure tokens 사이의 모든 방향을 통제 비교하는 설계
- original-unit inverse 선택, cell-cluster bootstrap, prediction-file 재현, data/split/preprocessing/config hash를 결합한 평가 계약
- NTL·발전소 construction overlap을 명시적으로 제거하는 ablation과 local-inventory 한계를 함께 보고하는 설계
- cell-fixed primary evaluation과 empirical-autocorrelation-based buffered robustness를 분리하는 프로토콜

관련 novelty를 주장하려면 Phase 4 결과 후 검색 범위·검색일·query·포함/제외 기준을 문서화한 별도 prior-art search가 필요하다.

### 11.4 필수 baseline 및 ablation 실험

1. train mean
2. raw-target LightGBM
3. log1p-target LightGBM: median inverse와 train-only Duan
4. Tweedie LightGBM
5. concat MLP
6. token encoder + no cross-attention
7. `infrastructure <- pollution/environment`
8. reverse `pollution/environment <- infrastructure`
9. bidirectional
10. NTL 제거
11. power-plant count/capacity 제거
12. NTL+power-plant 동시 제거
13. raw/log1p/Tweedie 및 median/Duan sensitivity
14. cell-fixed primary와 data-driven buffered spatial robustness
15. 가능하면 `v1_legacy`와 mass-preserving `v2_corrected` 비교
16. 가능하면 Delhi local inventory 또는 독립 top-down 관측 외부검증

모든 neural ablation은 tokenizer·hidden size·parameter count·optimizer·train budget·validation selection을 통제하고 여러 train seed의 cell-cluster CI를 보고해야 한다.

### 11.5 `IMPLEMENTATION_SPEC.md`에 필요한 변경 제안

이번 단계에서는 파일을 수정하지 않았다. 다음 내용을 Phase 3.6에서 제안 patch로 반영해야 한다.

1. target 정의를 `v1_legacy ODIAC-derived zonal mean`으로 고정하고 물리적 tC total 주장을 금지한다.
2. ODIAC2025 exact file manifest·checksum·header 없이는 release를 확정하지 않는 조건을 추가한다.
3. `v2_corrected`에 area-weighted mass conservation equation과 경계 정의·tC/tCO2 변환을 추가한다.
4. legacy GEE 결측치가 split 전 월 전체 통계로 채워졌다는 upstream leakage를 명시하고 raw pre-imputation feature 재구축을 요구한다.
5. NTL·power-plant construction overlap과 필수 feature ablation을 P0로 승격한다.
6. stream 이름을 actual feature order와 통일한다: 3 pollution/environment, 4 socio-infrastructure. 반대 표기를 제거한다.
7. FT-Transformer는 numeric token precedent, TabTransformer는 categorical contextualization precedent로 범위를 구분한다.
8. cross-attention 방향을 H로 명시하고 concat/no-CA/reverse/bidirectional을 필수로 한다.
9. block/buffer를 empirical autocorrelation range로 정하는 protocol과 여러 origin sensitivity를 추가한다.
10. true temporal holdout 없이는 forecasting 용어를 금지한다.
11. primary estimand(mean 또는 median), selection metric, inverse rule, target transform scale의 물리 단위를 사전 선언한다.
12. original-unit headline, log/Spearman secondary를 유지하고 local-inventory external validity 제한을 추가한다.
13. related work에 Bilotta 2025뿐 아니라 Yang 2019, Mustafa 2021, Zhang 2022/2024, Ou 2026과 inventory-comparison 연구를 포함한다.

### 11.6 Phase 4 전 blocking questions

1. 현재 NPZ를 만든 정확한 36개 ODIAC 파일은 무엇이며 checksum·header를 복구할 수 있는가?
2. 실제 사용 release가 ODIAC2025인가, 아니면 이전 release인데 파일명/설정만 바뀐 것인가?
3. target cell은 full 1 km square인가, Delhi rectangle으로 clipped된 polygon인가?
4. 논문 보고 단위는 tC/cell/month인가 tCO2/cell/month인가?
5. raw, pre-imputation GEE extraction을 재생성할 credential·asset revision·quality mask가 있는가?
6. project의 stream 명칭은 3 pollution/environment + 4 socio-infrastructure로 확정하는가?
7. 주 estimand는 conditional median/MAE인가 conditional mean/총량인가?
8. 실제 target의 zero mass, variance-mean 관계, 이분산성은 어떠한가?
9. empirical spatial autocorrelation range와 anisotropy는 몇 km인가?
10. Delhi local inventory 또는 독립적인 external validation 자료를 확보할 수 있는가?
11. 연구를 forecasting으로 확장할 것인가? 그렇다면 별도 future-month holdout을 만들 것인가?

1~5가 해결되지 않으면 `v1_legacy`를 물리적으로 해석한 Phase 4 본실험을 시작하지 않는 것이 타당하다. architecture smoke는 가능하지만 paper result로 동결하면 안 된다.

### 11.7 우선순위 Phase 3.6 구현 체크리스트

| 우선순위 | 작업 | 완료 조건 |
|---|---|---|
| P0-1 | ODIAC source provenance 복구 | 36개 파일명, release, checksum, CRS/transform/unit/nodata manifest |
| P0-2 | target aggregation 검증 | synthetic alignment tests, area-weighted aggregation, regional conservation test, boundary policy |
| P0-3 | `v2_corrected` 생성 계획 | immutable output path와 새 data hash; `v1_legacy` 미덮어쓰기 |
| P0-4 | upstream imputation 제거 | raw feature에서 split 후 train-only fit; GEE revision/QA metadata 기록 |
| P0-5 | 연구 claim/spec 수정안 | 본 감사의 11.5 항목을 검토 가능한 patch로 제시 |
| P1-1 | feature-overlap ablation 경로 | NTL, power, both 제거 candidate가 동일 evaluator 사용 |
| P1-2 | spatial diagnostic | target/predictor variogram, anisotropy, candidate block/buffer table |
| P1-3 | spatial split 동결안 | multiple origin sensitivity, pairwise minimum distance, dropped/domain coverage 보고 |
| P1-4 | architecture contract | actual stream naming, numeric tokenizer, no-CA/reverse/bidirectional, matched capacity |
| P1-5 | estimand 사전 선언 | mean/median, primary metric, log scale, inverse selection, Duan assumptions 기록 |
| P2-1 | external validity 계획 | Delhi/local inventory 또는 top-down 자료 접근성·공간/부문 경계 매핑 |
| P2-2 | documented prior-art search | query, database, date, inclusion criteria, negative findings까지 기록; 이후에만 novelty 문구 검토 |

## 12. 최종 감사 판정

Cell-MSCA 구현 자체의 방법론적 기반은 존재하지만, 특정 stream 분할과 attention 방향은 검증되지 않은 설계 가설이다. 가장 큰 연구 리스크는 architecture가 아니라 target provenance·질량보존 집계·construction-proxy overlap·legacy upstream imputation이다. 따라서 Phase 4 전에 target과 데이터 계보를 P0로 해결하고, architecture 효과는 동일 baseline·방향·overlap·spatial robustness ablation으로 분리해야 한다.
