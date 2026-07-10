;
; Input signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
;
; Output signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
; shader hash: 62a71a5bc234e861f60c60ed2e85d2cb
;
; Pipeline Runtime Information: 
;
;PSVRuntimeInfo:
; Compute Shader
; NumThreads=(32,1,1)
; NumBytesGroupSharedMemory: 0
; MinimumExpectedWaveLaneCount: 0
; MaximumExpectedWaveLaneCount: 4294967295
; UsesViewID: false
; SigInputElements: 0
; SigOutputElements: 0
; SigPatchConstOrPrimElements: 0
; SigInputVectors: 0
; SigOutputVectors[0]: 0
; SigOutputVectors[1]: 0
; SigOutputVectors[2]: 0
; SigOutputVectors[3]: 0
; EntryFunctionName: main
;
;
; Buffer Definitions:
;
; cbuffer params
; {
;
;   struct params
;   {
;
;       struct struct.DXLAWaveGEMMParams
;       {
;
;           uint M;                                   ; Offset:    0
;           uint N;                                   ; Offset:    4
;           uint K;                                   ; Offset:    8
;           uint stride_a;                            ; Offset:   12
;           uint stride_b;                            ; Offset:   16
;           uint stride_c;                            ; Offset:   20
;           uint transposed_b;                        ; Offset:   24
;           uint wave_size;                           ; Offset:   28
;           uint reserved[9];                         ; Offset:   32
;       
;       } params;                                     ; Offset:    0
;
;   
;   } params;                                         ; Offset:    0 Size:   164
;
; }
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
; params                            cbuffer      NA          NA     CB0            cb0     1
; matrix_a                          texture    byte         r/o      T0             t0     1
; matrix_b                          texture    byte         r/o      T1             t1     1
; result                                UAV    byte         r/w      U0             u0     1
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.ResBind = type { i32, i32, i32, i8 }
%dx.types.ResourceProperties = type { i32, i32 }
%dx.types.LinAlgMatrixC9M16N16U2S1 = type { i8* }
%dx.types.LinAlgMatrixC8M16N16U1S1 = type { i8* }
%dx.types.LinAlgMatrixC8M16N16U0S1 = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%struct.ByteAddressBuffer = type { i32 }
%struct.RWByteAddressBuffer = type { i32 }
%params = type { %struct.DXLAWaveGEMMParams }
%struct.DXLAWaveGEMMParams = type { i32, i32, i32, i32, i32, i32, i32, i32, [9 x i32] }

@dx.nothing.a = internal constant [1 x i32] zeroinitializer

define void @main() {
  %1 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 0, i32 0, i8 1 }, i32 0, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 1, i32 1, i32 0, i8 0 }, i32 1, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind zeroinitializer, i32 0, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %4 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 0, i32 0, i8 2 }, i32 0, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %5 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %4, %dx.types.ResourceProperties { i32 13, i32 164 })  ; AnnotateHandle(res,props)  resource: CBuffer
  %6 = call i32 @dx.op.groupId.i32(i32 94, i32 0)  ; GroupId(component)
  %7 = call i32 @dx.op.groupId.i32(i32 94, i32 1)  ; GroupId(component)
  %8 = alloca %dx.types.LinAlgMatrixC9M16N16U2S1
  %9 = alloca %dx.types.LinAlgMatrixC8M16N16U1S1
  %10 = alloca %dx.types.LinAlgMatrixC9M16N16U2S1
  %11 = alloca %dx.types.LinAlgMatrixC8M16N16U0S1
  %12 = mul i32 %7, 16
  %13 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %14 = mul i32 %6, 16
  %15 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %16 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %17 = extractvalue %dx.types.CBufRet.i32 %16, 0
  %18 = icmp uge i32 %12, %17
  %19 = icmp ne i1 %18, false
  %20 = icmp ne i1 %19, false
  br i1 %20, label %27, label %21

; <label>:21                                      ; preds = %0
  %22 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %23 = extractvalue %dx.types.CBufRet.i32 %22, 1
  %24 = icmp uge i32 %14, %23
  %25 = icmp ne i1 %24, false
  %26 = icmp ne i1 %25, false
  br i1 %26, label %27, label %28

; <label>:27                                      ; preds = %21, %0
  br label %115

; <label>:28                                      ; preds = %21
  %29 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %30 = call %dx.types.LinAlgMatrixC9M16N16U2S1 @dx.op.linAlgFillMatrix.mC9M16N16U2S1.f32(i32 -2147483636, float 0.000000e+00)  ; LinAlgFillMatrix(value)
  store %dx.types.LinAlgMatrixC9M16N16U2S1 %30, %dx.types.LinAlgMatrixC9M16N16U2S1* %8
  %31 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %32 = load %dx.types.LinAlgMatrixC9M16N16U2S1, %dx.types.LinAlgMatrixC9M16N16U2S1* %8
  store %dx.types.LinAlgMatrixC9M16N16U2S1 %32, %dx.types.LinAlgMatrixC9M16N16U2S1* %10
  %33 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %34 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %35 = extractvalue %dx.types.CBufRet.i32 %34, 2
  %36 = icmp ult i32 0, %35
  br i1 %36, label %37, label %100

; <label>:37                                      ; preds = %28
  br label %38

; <label>:38                                      ; preds = %91, %37
  %39 = phi i32 [ 0, %37 ], [ %92, %91 ]
  %40 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %41 = extractvalue %dx.types.CBufRet.i32 %40, 3
  %42 = mul i32 %12, %41
  %43 = add i32 %42, %39
  %44 = mul i32 %43, 2
  %45 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %46 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %47 = extractvalue %dx.types.CBufRet.i32 %46, 2
  %48 = icmp ne i32 %47, 0
  br i1 %48, label %49, label %55

; <label>:49                                      ; preds = %38
  %50 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %51 = extractvalue %dx.types.CBufRet.i32 %50, 0
  %52 = mul i32 %14, %51
  %53 = add i32 %52, %39
  %54 = mul i32 %53, 2
  br label %61

; <label>:55                                      ; preds = %38
  %56 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %57 = extractvalue %dx.types.CBufRet.i32 %56, 0
  %58 = mul i32 %39, %57
  %59 = add i32 %58, %14
  %60 = mul i32 %59, 2
  br label %61

; <label>:61                                      ; preds = %55, %49
  %62 = phi i32 [ %54, %49 ], [ %60, %55 ]
  %63 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %64 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %65 = extractvalue %dx.types.CBufRet.i32 %64, 3
  %66 = mul i32 %65, 2
  %67 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %68 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %3, %dx.types.ResourceProperties { i32 11, i32 0 })  ; AnnotateHandle(res,props)  resource: ByteAddressBuffer
  %69 = call %dx.types.LinAlgMatrixC8M16N16U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.mC8M16N16U0S1(i32 -2147483634, %dx.types.Handle %68, i32 %44, i32 %66, i32 0, i32 128)  ; LinAlgMatrixLoadFromDescriptor(handle,offset,stride,layout,align)
  store %dx.types.LinAlgMatrixC8M16N16U0S1 %69, %dx.types.LinAlgMatrixC8M16N16U0S1* %11
  %70 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %71 = load %dx.types.LinAlgMatrixC8M16N16U0S1, %dx.types.LinAlgMatrixC8M16N16U0S1* %11
  %72 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %73 = extractvalue %dx.types.CBufRet.i32 %72, 2
  %74 = icmp ne i32 %73, 0
  %75 = select i1 %74, i32 1, i32 0
  %76 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %77 = extractvalue %dx.types.CBufRet.i32 %76, 0
  %78 = mul i32 %77, 2
  %79 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %80 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %2, %dx.types.ResourceProperties { i32 11, i32 0 })  ; AnnotateHandle(res,props)  resource: ByteAddressBuffer
  %81 = call %dx.types.LinAlgMatrixC8M16N16U1S1 @dx.op.linAlgMatrixLoadFromDescriptor.mC8M16N16U1S1(i32 -2147483634, %dx.types.Handle %80, i32 %62, i32 %78, i32 %75, i32 128)  ; LinAlgMatrixLoadFromDescriptor(handle,offset,stride,layout,align)
  store %dx.types.LinAlgMatrixC8M16N16U1S1 %81, %dx.types.LinAlgMatrixC8M16N16U1S1* %9
  %82 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %83 = load %dx.types.LinAlgMatrixC8M16N16U1S1, %dx.types.LinAlgMatrixC8M16N16U1S1* %9
  %84 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %85 = load %dx.types.LinAlgMatrixC9M16N16U2S1, %dx.types.LinAlgMatrixC9M16N16U2S1* %10, align 4
  %86 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %87 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %88 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %89 = call %dx.types.LinAlgMatrixC9M16N16U2S1 @dx.op.linAlgMatrixMultiplyAccumulate.mC9M16N16U2S1.mC8M16N16U0S1.mC8M16N16U1S1.mC9M16N16U2S1(i32 -2147483637, %dx.types.LinAlgMatrixC8M16N16U0S1 %71, %dx.types.LinAlgMatrixC8M16N16U1S1 %83, %dx.types.LinAlgMatrixC9M16N16U2S1 %85)  ; LinAlgMatrixMultiplyAccumulate(matrixA,matrixB,matrixC)
  store %dx.types.LinAlgMatrixC9M16N16U2S1 %89, %dx.types.LinAlgMatrixC9M16N16U2S1* %10
  %90 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  br label %91

; <label>:91                                      ; preds = %61
  %92 = add i32 %39, 16
  %93 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %94 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %95 = extractvalue %dx.types.CBufRet.i32 %94, 2
  %96 = icmp ult i32 %92, %95
  %97 = icmp ne i1 %96, false
  %98 = icmp ne i1 %97, false
  br i1 %98, label %38, label %99

; <label>:99                                      ; preds = %91
  br label %100

; <label>:100                                     ; preds = %99, %28
  %101 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %102 = extractvalue %dx.types.CBufRet.i32 %101, 1
  %103 = mul i32 %12, %102
  %104 = add i32 %103, %14
  %105 = mul i32 %104, 4
  %106 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %107 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %108 = extractvalue %dx.types.CBufRet.i32 %107, 1
  %109 = mul i32 %108, 4
  %110 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %111 = load %dx.types.LinAlgMatrixC9M16N16U2S1, %dx.types.LinAlgMatrixC9M16N16U2S1* %10, align 4
  %112 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  %113 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %1, %dx.types.ResourceProperties { i32 4107, i32 0 })  ; AnnotateHandle(res,props)  resource: RWByteAddressBuffer
  call void @dx.op.linAlgMatrixStoreToDescriptor.mC9M16N16U2S1(i32 -2147483628, %dx.types.LinAlgMatrixC9M16N16U2S1 %111, %dx.types.Handle %113, i32 %105, i32 %109, i32 0, i32 128)  ; LinAlgMatrixStoreToDescriptor(matrix,handle,offset,stride,layout,align)
  %114 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  br label %115

; <label>:115                                     ; preds = %100, %27
  %116 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.groupId.i32(i32, i32) #0

; Function Attrs: nounwind
declare %dx.types.LinAlgMatrixC9M16N16U2S1 @dx.op.linAlgFillMatrix.mC9M16N16U2S1.f32(i32, float) #1

; Function Attrs: nounwind
declare %dx.types.LinAlgMatrixC8M16N16U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.mC8M16N16U0S1(i32, %dx.types.Handle, i32, i32, i32, i32) #1

; Function Attrs: nounwind
declare %dx.types.LinAlgMatrixC8M16N16U1S1 @dx.op.linAlgMatrixLoadFromDescriptor.mC8M16N16U1S1(i32, %dx.types.Handle, i32, i32, i32, i32) #1

; Function Attrs: nounwind
declare %dx.types.LinAlgMatrixC9M16N16U2S1 @dx.op.linAlgMatrixMultiplyAccumulate.mC9M16N16U2S1.mC8M16N16U0S1.mC8M16N16U1S1.mC9M16N16U2S1(i32, %dx.types.LinAlgMatrixC8M16N16U0S1, %dx.types.LinAlgMatrixC8M16N16U1S1, %dx.types.LinAlgMatrixC9M16N16U2S1) #1

; Function Attrs: nounwind
declare void @dx.op.linAlgMatrixStoreToDescriptor.mC9M16N16U2S1(i32, %dx.types.LinAlgMatrixC9M16N16U2S1, %dx.types.Handle, i32, i32, i32, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #2

; Function Attrs: nounwind readnone
declare %dx.types.Handle @dx.op.annotateHandle(i32, %dx.types.Handle, %dx.types.ResourceProperties) #0

; Function Attrs: nounwind readnone
declare %dx.types.Handle @dx.op.createHandleFromBinding(i32, %dx.types.ResBind, i32, i1) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }

!dx.targetTypes = !{!0, !1, !2}
!llvm.ident = !{!3}
!dx.version = !{!4}
!dx.valver = !{!4}
!dx.shaderModel = !{!5}
!dx.resources = !{!6}
!dx.entryPoints = !{!14}

!0 = !{%dx.types.LinAlgMatrixC9M16N16U2S1 undef, i32 9, i32 16, i32 16, i32 2, i32 1}
!1 = !{%dx.types.LinAlgMatrixC8M16N16U0S1 undef, i32 8, i32 16, i32 16, i32 0, i32 1}
!2 = !{%dx.types.LinAlgMatrixC8M16N16U1S1 undef, i32 8, i32 16, i32 16, i32 1, i32 1}
!3 = !{!"dxcoob 1.10.2605.24 (c1e1fc784)"}
!4 = !{i32 1, i32 10}
!5 = !{!"cs", i32 6, i32 10}
!6 = !{!7, !10, !12, null}
!7 = !{!8, !9}
!8 = !{i32 0, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!9 = !{i32 1, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 1, i32 1, i32 11, i32 0, null}
!10 = !{!11}
!11 = !{i32 0, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!12 = !{!13}
!13 = !{i32 0, %params* undef, !"", i32 0, i32 0, i32 1, i32 164, null}
!14 = !{void ()* @main, !"main", null, !6, !15}
!15 = !{i32 0, i64 8598323216, i32 4, !16}
!16 = !{i32 32, i32 1, i32 1}
