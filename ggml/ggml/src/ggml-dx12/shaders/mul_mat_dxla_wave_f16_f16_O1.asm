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
; shader hash: f4052967fdd4df629c19f7fca5b1ad30
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
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.LinAlgMatrixC9M16N16U2S1 = type { i8* }
%dx.types.LinAlgMatrixC8M16N16U0S1 = type { i8* }
%dx.types.LinAlgMatrixC8M16N16U1S1 = type { i8* }
%struct.ByteAddressBuffer = type { i32 }
%struct.RWByteAddressBuffer = type { i32 }
%params = type { %struct.DXLAWaveGEMMParams }
%struct.DXLAWaveGEMMParams = type { i32, i32, i32, i32, i32, i32, i32, i32, [9 x i32] }

define void @main() {
  %1 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 0, i32 0, i8 1 }, i32 0, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 1, i32 1, i32 0, i8 0 }, i32 1, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind zeroinitializer, i32 0, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %4 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 0, i32 0, i8 2 }, i32 0, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %5 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %4, %dx.types.ResourceProperties { i32 13, i32 164 })  ; AnnotateHandle(res,props)  resource: CBuffer
  %6 = call i32 @dx.op.groupId.i32(i32 94, i32 0)  ; GroupId(component)
  %7 = call i32 @dx.op.groupId.i32(i32 94, i32 1)  ; GroupId(component)
  %8 = shl i32 %7, 4
  %9 = shl i32 %6, 4
  %10 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %11 = extractvalue %dx.types.CBufRet.i32 %10, 0
  %12 = icmp ult i32 %8, %11
  %13 = extractvalue %dx.types.CBufRet.i32 %10, 1
  %14 = icmp ult i32 %9, %13
  %15 = and i1 %12, %14
  br i1 %15, label %16, label %68

; <label>:16                                      ; preds = %0
  %17 = call %dx.types.LinAlgMatrixC9M16N16U2S1 @dx.op.linAlgFillMatrix.mC9M16N16U2S1.f32(i32 -2147483636, float 0.000000e+00)  ; LinAlgFillMatrix(value)
  %18 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %19 = extractvalue %dx.types.CBufRet.i32 %18, 2
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %21, label %30

; <label>:21                                      ; preds = %49, %16
  %22 = phi %dx.types.LinAlgMatrixC9M16N16U2S1 [ %17, %16 ], [ %63, %49 ]
  %23 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %24 = extractvalue %dx.types.CBufRet.i32 %23, 1
  %25 = mul i32 %24, %8
  %26 = add i32 %25, %9
  %27 = shl i32 %26, 2
  %28 = shl i32 %24, 2
  %29 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %1, %dx.types.ResourceProperties { i32 4107, i32 0 })  ; AnnotateHandle(res,props)  resource: RWByteAddressBuffer
  call void @dx.op.linAlgMatrixStoreToDescriptor.mC9M16N16U2S1(i32 -2147483628, %dx.types.LinAlgMatrixC9M16N16U2S1 %22, %dx.types.Handle %29, i32 %27, i32 %28, i32 0, i32 128)  ; LinAlgMatrixStoreToDescriptor(matrix,handle,offset,stride,layout,align)
  br label %68

; <label>:30                                      ; preds = %49, %16
  %31 = phi %dx.types.LinAlgMatrixC9M16N16U2S1 [ %63, %49 ], [ %17, %16 ]
  %32 = phi i32 [ %64, %49 ], [ 0, %16 ]
  %33 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %34 = extractvalue %dx.types.CBufRet.i32 %33, 3
  %35 = mul i32 %34, %8
  %36 = add i32 %35, %32
  %37 = shl i32 %36, 1
  %38 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %39 = extractvalue %dx.types.CBufRet.i32 %38, 2
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %45, label %41

; <label>:41                                      ; preds = %30
  %42 = extractvalue %dx.types.CBufRet.i32 %38, 0
  %43 = mul i32 %42, %9
  %44 = add i32 %43, %32
  br label %49

; <label>:45                                      ; preds = %30
  %46 = extractvalue %dx.types.CBufRet.i32 %38, 0
  %47 = mul i32 %46, %32
  %48 = add i32 %47, %9
  br label %49

; <label>:49                                      ; preds = %45, %41
  %50 = phi i32 [ %44, %41 ], [ %48, %45 ]
  %51 = shl i32 %50, 1
  %52 = shl i32 %34, 1
  %53 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %3, %dx.types.ResourceProperties { i32 11, i32 0 })  ; AnnotateHandle(res,props)  resource: ByteAddressBuffer
  %54 = call %dx.types.LinAlgMatrixC8M16N16U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.mC8M16N16U0S1(i32 -2147483634, %dx.types.Handle %53, i32 %37, i32 %52, i32 0, i32 128)  ; LinAlgMatrixLoadFromDescriptor(handle,offset,stride,layout,align)
  %55 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %56 = extractvalue %dx.types.CBufRet.i32 %55, 2
  %57 = icmp ne i32 %56, 0
  %58 = zext i1 %57 to i32
  %59 = extractvalue %dx.types.CBufRet.i32 %55, 0
  %60 = shl i32 %59, 1
  %61 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %2, %dx.types.ResourceProperties { i32 11, i32 0 })  ; AnnotateHandle(res,props)  resource: ByteAddressBuffer
  %62 = call %dx.types.LinAlgMatrixC8M16N16U1S1 @dx.op.linAlgMatrixLoadFromDescriptor.mC8M16N16U1S1(i32 -2147483634, %dx.types.Handle %61, i32 %51, i32 %60, i32 %58, i32 128)  ; LinAlgMatrixLoadFromDescriptor(handle,offset,stride,layout,align)
  %63 = call %dx.types.LinAlgMatrixC9M16N16U2S1 @dx.op.linAlgMatrixMultiplyAccumulate.mC9M16N16U2S1.mC8M16N16U0S1.mC8M16N16U1S1.mC9M16N16U2S1(i32 -2147483637, %dx.types.LinAlgMatrixC8M16N16U0S1 %54, %dx.types.LinAlgMatrixC8M16N16U1S1 %62, %dx.types.LinAlgMatrixC9M16N16U2S1 %31)  ; LinAlgMatrixMultiplyAccumulate(matrixA,matrixB,matrixC)
  %64 = add i32 %32, 16
  %65 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %5, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %66 = extractvalue %dx.types.CBufRet.i32 %65, 2
  %67 = icmp ult i32 %64, %66
  br i1 %67, label %30, label %21

; <label>:68                                      ; preds = %21, %0
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
