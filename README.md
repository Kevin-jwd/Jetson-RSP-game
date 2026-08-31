# Jetson RPS Game (가위바위보 게임)

Jetson Orin Nano의 TensorRT 엔진 기반으로 동작하며, 웹캠을 통해 실시간으로 가위바위보를 플레이하는 게임 프로젝트입니다. 타이틀 화면에서 두 가지 모드를 제공합니다:

- **1인용** — 카메라 좌측에 위치한 사용자의 손 1개와 AI의 무작위 출제 동작이 대결합니다.
  오락기처럼 크레딧(코인) 시스템으로 동작합니다. 크레딧을 소모하여 판에 참여하며, **패배했을 때만** 크레딧이 1개 차감됩니다. 따라서 승리하거나 비기면 계속 게임을 이어서 플레이할 수 있습니다. `c` 키를 눌러 코인을 투입할 수 있으며, 크레딧이 0이 되면 게임 오버(GAME OVER) 상태가 되어 타이틀 화면에서 추가 코인을 요구합니다.
- **2인용** — 화면에 포착된 두 손을 좌우로 구분하여 승패를 판정합니다. 한 판을 진행한 후 재시도 또는 종료를 선택할 수 있습니다.

게임은 "가위-바위-보" 구령에 맞춰 0.4초간 손 모양을 관찰하고 다수결 투표(Majority vote)로 최종 동작을 판정한 뒤, 해당 판정 결과 프레임을 멈춰서(Freeze) 보여줍니다. `q` 키를 눌러야만 프로그램을 종료할 수 있습니다.

## 실행 방법 (Run)

```bash
pip install -r requirements.txt
python main.py --model models/rps_yolo11s.engine
```

옵션: `--model`, `--camera`, `--conf`, `--classes`, `--no-mirror`, `--no-flip-tta`
조작키: `c` 코인 투입, `r` 재시도, `m` 좌우 반전(Mirror), `q` 종료 (유일한 종료 방법). 시스템에 한글 폰트가 설치되어 있지 않은 경우 텍스트 라벨은 영문으로 대체 출력됩니다.

## 프로젝트 구조 (Layout)

| 경로 | 역할 |
| --- | --- |
| `main.py` | 커맨드라인 인자 파싱 |
| `rps/detector.py` | TensorRT 엔진 기반 YOLO11 추론 (`detect(frame) -> [Detection]` 인터페이스 제공) |
| `rps/cuda.py` | 디바이스 메모리 관리 (`cuda-python` 또는 `pycuda` 대응) |
| `rps/logic.py` | 승/패/무승부 판정 규칙 |
| `rps/app.py` | 라운드 상태 머신 (메뉴 → 카운트다운 → 슛 → 결과 판정), 버튼 및 화면 렌더링 |
| `rps/particles.py` | 승리한 손 위치에 터지는 파티클 이펙트 |
| `rps/retro.py` | 레트로 아케이드 연출: 픽셀 텍스트, 스캔라인, 원근감, 외각선 테두리 처리 |
| `assets/` | AI 핸드 렌더링용 `rock/paper/scissors` 이미지 파일 (선택 사항) |
| `test/preview.py` | 바운딩 박스, 신뢰도(Confidence), FPS를 확인하는 라이브 카메라 프레임 미리보기 |
| `tools/probe.py` | 엔진이 반환하는 출력을 GUI 없이 확인하는 디버깅 툴 |
| `tools/cuda_check.py` | CUDA 및 TensorRT 환경 점검 스크립트 |
| `tools/merge_dataset.py` | 클래스 이름 기준으로 ID를 재매핑하여 YOLO 데이터셋을 통합하는 스크립트 |
| `models/rps_yolo11n.onnx` | 초기 검출기 모델 (320x320) — 보드에서 엔진 생성용 소스로 보관 |

각 모듈의 세부 동작 원리 및 개발 의도는 `docs/ARCHITECTURE.md` 문서에서 확인하실 수 있습니다.

## 승패 판정 방식 (Judging)

단일 프레임만으로 판정하는 것은 오판 위험이 큽니다. "가위바위보" 구령이 끝나는 시점에도 손은 여전히 움직이는 중일 수 있으며, 주먹을 펴서 보를 만드는 과정의 중간 동작이 가위로 잘못 인식될 수 있기 때문입니다.

따라서 본 게임은 `VOTE_MS` (0.4초) 동안 손 하나당 프레임별로 1표씩 라벨을 수집한 뒤 다수결(Majority)로 최종 동작을 결정합니다. 동점이 발생할 경우 신뢰도 합산(Summed confidence)이 더 높은 쪽을 선택합니다. 또한 화면 속의 손은 라벨이 아닌 바운딩 박스의 좌우 위치를 기준으로 플레이어를 식별합니다.

## 오른손/왼손 보정 (Handedness)

단일 방향 손 데이터셋으로 학습된 모델은 왼손과 오른손에 대해 편향성을 갖기 쉽습니다. 동일한 손짓이라도 왼손인지 오른손인지에 따라 인식 점수가 달라지므로, 카메라 방향에 따라 한쪽 플레이어가 불리해질 수 있습니다.

이를 해결하기 위해 각 프레임을 원본 그대로 한 번, 좌우 반전(Mirrored)하여 한 번 총 두 번 추론을 수행한 뒤 각 손마다 더 높은 신뢰도를 얻은 결과를 채택합니다(TTA 기법). `--no-flip-tta` 옵션을 사용하면 이 기능을 끄고 추론 연산 비용을 절반으로 줄일 수 있습니다.

참고로 화면상의 좌우 반전(Mirror) 옵션은 단순 출력용이며, 모델 추론 파이프라인에는 영향을 주지 않습니다.

## 추론 실행 시점 (When inference runs)

모델 추론은 라운드의 승패를 결정하는 핵심 순간에만 동작합니다. 구령이 나오는 동안, 타이틀 화면, 그리고 결과가 멈춰있는 화면에서는 승패를 결정할 필요가 없으므로 모델을 실행하는 것은 프레임 자원 낭비입니다.

따라서 판정 결과가 나오기 전까지는 화면에 바운딩 박스가 표시되지 않습니다. 손이 화면 프레임 내에 제대로 들어오는지 확인하려면 디버깅용 미리보기 스크립트를 실행하세요:

```bash
python test/preview.py --model models/rps_yolo11s.engine
```

## 클래스 순서 주의사항 (Class order)

클래스 순서는 잘못 설정되기 쉽고 오류 메시지 없이 조용히 실패합니다. 바위(Rock)를 보(Paper)로 잘못 인식하는 경우, 코드 버그가 아니라 단순히 모델 성능이 떨어지는 것처럼 보일 수 있습니다. 흔히 사용되는 클래스 순서들 간에는 주로 0번과 2번 인덱스만 차이가 나므로, **바위(Rock)는 정상 인식되지만 보(Paper)와 가위(Scissors)가 서로 바뀌어 인식**되는 현상이 발생합니다. 이런 증상이 보이면 모델 정확도 대신 클래스 순서를 먼저 의심해야 합니다.

클래스 순서의 적용 우선순위는 다음과 같습니다:
1. `--classes` 인자값
2. 엔진 내부 JSON 헤더 (Ultralytics 내보내기 모델)
3. `rps/detector.py`에 정의된 기본값 (`CLASS_NAMES`, Roboflow 데이터셋 순서: `paper, rock, scissors`)

구버전 클래스 순서로 생성된 엔진을 사용할 경우 `--classes scissors,rock,paper` 옵션을 명시해야 합니다.

## Jetson 환경 설정 (Jetson)

본 프로젝트의 추론 로직은 JetPack에 기본 포함된 `tensorrt` 모듈을 직접 호출하므로, Jetson 보드 상에 PyTorch나 Ultralytics를 별도로 설치할 필요가 없습니다. 디바이스 메모리 관리를 위한 `cuda-python` (또는 `pycuda`)만 설치해주면 됩니다:

```bash
pip install cuda-python
```

TensorRT 엔진 파일(`.engine`)은 빌드된 환경의 GPU 및 TensorRT 버전과 종속성이 존재하므로, **반드시 실행할 Jetson 보드 위에서 직접 생성**해야 합니다. 기존에 플레이 중인 엔진 파일을 덮어쓰면 복구할 백업이 사라지므로, 빌드 시마다 서로 다른 이름을 부여하는 것이 좋습니다:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=models/rps_yolo11s.engine --fp16
```

손이 전혀 감지되지 않는다면 게임을 실행하기 전에 엔진 파일부터 점검해 보세요. `tools/probe.py` 스크립트를 사용하면 GUI 없이 엔진의 입출력 Shape, 클래스 이름, 낮은 임계값에서의 원시 점수(Raw score)를 출력해 볼 수 있습니다:

```bash
python tools/probe.py --model models/rps_yolo11s.engine
```

`tools/cuda_check.py` 스크립트는 현재 프로세스가 인식하는 TensorRT 및 CUDA 버전을 출력해 주므로, 코드 문제와 환경 설정 문제를 구별하여 진단하는 데 유용합니다.