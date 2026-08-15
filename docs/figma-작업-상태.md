# Figma 작업 상태

파일: https://www.figma.com/design/gXdD3Oaq1DJmoou8R75znr
최종 갱신: 2026-08-16
중단 사유: **Figma MCP 호출 한도 (Starter 플랜)**

## 플랜 제약

Starter 플랜에서 두 번 막혔다. 다음 세션에서도 같은 벽에 부딪힌다.

| 제약 | 영향 |
|---|---|
| 페이지 3개 | "컴포넌트당 1페이지" 관례 불가. Section으로 대체 |
| MCP 호출 한도 | 한 세션에 20회 안팎. 컴포넌트 재작업을 못 끝냄 |

호출 한도가 실질적인 병목이다. 한 번의 `use_figma`에 최대한 담되,
스크립트가 실패하면 통째로 롤백되므로(atomic) 너무 키우면 위험하다.

## 완료

- 페이지 3개: `00 Foundations & Components` / `01 Screens` / `02 Flows`
- 변수 51개
  - Color 23개 — 대비 7:1 검증
  - Size 20개 — space, radius, touch, font
  - Config 8개 — confidence 임계값, 녹화·카운트다운 초, 고정 계약값
- Color·Size 43개에 WEB Code Syntax (`var(--text-primary)` 형태)
- 텍스트 스타일 8개 (Noto Sans KR)
- 화면 13개 — **도형 직접 그림, 컴포넌트 아님**
- Flows 4장 — WS↔화면 매핑, 신뢰도 분기, 오류 12종, 고정 계약
- Foundations 문서 — 팔레트, 타입 램프, 접근성 규칙
- `Button` 컴포넌트 세트 (Style=Primary/Secondary/Disabled, Label 프로퍼티)

## 남은 작업

순서대로 진행한다. 각 항목이 `use_figma` 1회 분량이다.

1. `StatusBadge` — Tone=Neutral/Success/Recording, Label·Icon 텍스트 프로퍼티
2. `ErrorBanner` — Tone=Warning/Danger, Message 프로퍼티
3. `CameraPreview` — State=Idle/Ready/Recording/Dimmed, 1088×612 고정
4. `ProgressRing` — 남은 시간 링. arcData 사용
5. `Modal` — Title·Body·Button 슬롯
6. 화면 13개를 인스턴스로 재조립 (3~4회로 분할)
7. 프로토타입 연결
8. 커버 페이지

## 재개 방법

```
docs/figma-작업-상태.md 를 읽고 Figma 작업을 이어서 해줘.
```

`use_figma` 호출 전 `figma-use` 스킬을 MCP 리소스로 읽어야 한다
(`skill://figma/figma-use/SKILL.md`). Claude Code 슬래시 스킬로는 없다.

## 현재 파일의 문제

화면이 컴포넌트를 쓰지 않는다. 버튼 하나를 고치려면 13곳을 손대야 한다.
6번 항목이 이 작업의 핵심이며, 아직 시작하지 않았다.
