"""
Solución para verificar las rutas de scispacy sin usar LinkerPaths (obsoleto)
"""
import os
from pathlib import Path

# Obtener el directorio home del usuario
home = Path.home()

# La ruta predeterminada donde scispacy guarda sus datos
scispacy_data_dir = home / ".scispacy" / "datasets"

print("📁 Rutas de datos de scispacy:\n")
print(f"   Directorio principal: {scispacy_data_dir}")
print(f"   Existe: {scispacy_data_dir.exists()}")

if scispacy_data_dir.exists():
    print(f"\n📊 Archivos descargados:")
    for file in scispacy_data_dir.iterdir():
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   • {file.name[:50]}... ({size_mb:.1f} MB)")
else:
    print("\n⚠️  El directorio de datos aún no existe. Se creará cuando descargues modelos con EntityLinker.")

# Alternativa: Verificar directamente desde el linker si ya lo has configurado
print("\n\n💡 Alternativa: Verificar desde el EntityLinker activo")
print("   Si ya cargaste el linker en tu código, usa:")
print("   >>> linker = nlp_scibert.get_pipe('scispacy_linker')")
print("   >>> print(f'Conceptos disponibles: {len(linker.kb.cui_to_entity)}')")
