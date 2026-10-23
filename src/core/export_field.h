#pragma once
// Cell-centred fields a scene can export (to .vdb, previews and slices).

namespace sim {

enum class ExportField {
    Density,      // fireball gas / smoke, mass concentration relative to ground air
    Dust,         // optical dust concentration
    Condensate,   // cloud water (white)
    Temperature,  // K, 0 where nothing is visible
    Flame,        // emission strength, (T / flame_temperature)^4 above ambient
    VelX,         // m/s, cell-centred, 0 where nothing is visible
    VelY,
    VelZ,
    Vorticity,    // |curl u| in 1/s, 0 where nothing is visible; drives render-time detail
    Count
};

inline const char* export_field_name(ExportField f) {
    switch (f) {
        case ExportField::Density: return "density";
        case ExportField::Dust: return "dust";
        case ExportField::Condensate: return "condensate";
        case ExportField::Temperature: return "temperature";
        case ExportField::Flame: return "flame";
        case ExportField::VelX: return "vel_x";
        case ExportField::VelY: return "vel_y";
        case ExportField::VelZ: return "vel_z";
        case ExportField::Vorticity: return "vorticity";
        default: return "?";
    }
}

}  // namespace sim
