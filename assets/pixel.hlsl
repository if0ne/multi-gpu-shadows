struct PixelInput
{
    float4 position : SV_POSITION;
};

float4 Main(PixelInput input) : SV_TARGET
{
    return float4(1.0, 1.0, 0.0, 1.0);
}