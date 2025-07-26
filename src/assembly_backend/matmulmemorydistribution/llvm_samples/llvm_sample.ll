; ModuleID = 'calculator'
source_filename = "calculator"

define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

define i32 @sub(i32 %a, i32 %b) {
entry:
  %diff = sub i32 %a, %b
  ret i32 %diff
}

define i32 @mul(i32 %a, i32 %b) {
entry:
  %prod = mul i32 %a, %b
  ret i32 %prod
}

define i32 @div(i32 %a, i32 %b) {
entry:
  %quot = sdiv i32 %a, %b
  ret i32 %quot
}

define i32 @mod(i32 %a, i32 %b) {
entry:
  %rem = srem i32 %a, %b
  ret i32 %rem
}

declare float @llvm.sin.f32(float)

define float @sine(float %x) {
entry:
  %s = call float @llvm.sin.f32(float %x)
  ret float %s
}

define i32 @main() {
entry:
  %a = call i32 @add(i32 10, i32 5)
  %b = call i32 @mul(i32 %a, i32 3)
  ret i32 %b
}
