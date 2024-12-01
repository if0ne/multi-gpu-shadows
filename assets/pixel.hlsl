Texture2D gDiffuseMap : register(t2);
Texture2D gNormalMap : register(t5);
SamplerState gsamLinearClamp : register(s0);

cbuffer Globals : register(b0)
{
    matrix ViewMatrix;
    matrix ProjectionMatrix;
    float3 EyePos;
}

cbuffer Mat : register(b1)
{
    float4 Diffuse;
    float FresnelR0;
    float Roughness;
}

cbuffer DirectionalLight : register(b3)
{
    float3 Strength;
    float3 Direction;
}

cbuffer AmbientLight : register(b4)
{
    float4 AmbientLight;
}

struct PixelInput
{
    float4 position : SV_POSITION;
    float3 world_pos: POSITION;
    float3 normal: NORMAL;
    float3 tangent: TANGENT;
    float3 bitangent : BITANGENT;
    float2 uv : TEXCOORD;
};

float3 SchlickFresnel(float3 R0, float3 normal, float3 lightVec)
{
    float cosIncidentAngle = saturate(dot(normal, lightVec));
    float f0 = 1.0f - cosIncidentAngle;
    float3 reflectPercent = R0 + (1.0f - R0)*(f0*f0*f0*f0*f0);

    return reflectPercent;
}

float3 BlinnPhong(float4 diffuseAlbedo, float3 lightStrength, float3 lightVec, float3 normal, float3 toEye)
{
    const float m = (1.0 - Roughness) * 256.0f;
    float3 halfVec = normalize(toEye + lightVec);

    float roughnessFactor = (m + 8.0f)*pow(max(dot(halfVec, normal), 0.0f), m) / 8.0f;
    float3 fresnelFactor = SchlickFresnel(FresnelR0, halfVec, lightVec);

    float3 specAlbedo = fresnelFactor*roughnessFactor;

    specAlbedo = specAlbedo / (specAlbedo + 1.0f);

    return (diffuseAlbedo.rgb + specAlbedo) * lightStrength;
}

float3 ComputeDirectionalLight(float4 diffuseAlbedo, float3 normal, float3 toEye)
{
    float3 lightVec = -Direction;

    float ndotl = max(dot(lightVec, normal), 0.0f);
    float3 lightStrength = Strength * ndotl;

    return BlinnPhong(diffuseAlbedo, lightStrength, lightVec, normal, toEye);
}

float4 Main(PixelInput input) : SV_TARGET
{
    float4 diffuseAlbedo = gDiffuseMap.Sample(gsamLinearClamp, input.uv) * Diffuse;

    float3 normalMap = gNormalMap.Sample(gsamLinearClamp, input.uv).rgb * 2.0 - 1.0;

    float3 T = normalize(input.tangent);
    float3 B = normalize(input.bitangent);
    float3 N = normalize(input.normal);
    float3x3 TBN = float3x3(T, B, N);
    input.normal = normalize(mul(normalMap, TBN));

    float3 toEye = normalize(EyePos - input.world_pos);
    float4 ambient = AmbientLight*diffuseAlbedo;

    float4 directLight = float4(ComputeDirectionalLight(diffuseAlbedo, input.normal, toEye), 1.0);
	float4 litColor = ambient + directLight;

	litColor.a = diffuseAlbedo.a;
    return litColor;
}