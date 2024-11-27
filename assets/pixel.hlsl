struct PixelInput
{
    float4 position : SV_POSITION;
    float3 normal: NORMAL;
};

float4 Main(PixelInput input) : SV_TARGET
{
    return float4(input.normal.x, input.normal.y, input.normal.z, 1.0);
}