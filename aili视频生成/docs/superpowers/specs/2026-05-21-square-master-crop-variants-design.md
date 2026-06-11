# Square Master Crop Variants Design

## Goal

Generate one new basketball video as a `1080P` `1:1` master, then derive nine aspect-ratio variants from that exact same video so the experiment has identical motion and content across sizes.

## Master Prompt

Use a prompt that keeps the basketball player smaller, centered, and fully visible, with extra surrounding space for later crop variants:

`一名篮球运动员在室内球馆完成高速快攻上篮，主体人物全身可见，人物占画面较小并位于画面中心，四周留出充足空间，适合后期裁切成横屏、竖屏、方形和超宽画幅，电影感灯光，真实运动摄影风格，稳定跟随镜头。`

## Variant Strategy

The master is generated once at `1080P` and `1:1`. All experimental variants are then created locally with FFmpeg from that master. Crops are center crops, which prioritizes the centered subject. Each crop is scaled to a standard target size.

## Output Variants

- `16:9`: `1920x1080`
- `9:16`: `1080x1920`
- `1:1`: `1080x1080`
- `4:3`: `1440x1080`
- `3:4`: `1080x1440`
- `4:5`: `1080x1350`
- `5:4`: `1350x1080`
- `9:21`: `1080x2520`
- `21:9`: `2520x1080`

## Notes

Extreme crops such as `9:21` and `21:9` necessarily discard much of a square frame. The prompt reduces this risk by placing the athlete smaller and centered.

