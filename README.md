# NukeSim

Симуляция наземного ядерного взрыва (C++20 / CUDA) с экспортом в OpenVDB для рендера в Blender.

План и архитектура: [docs/PLAN.md](docs/PLAN.md).

**Текущий этап: M0, каркас.** Сборка, бэкенд CPU/CUDA из одного исходника, экспорт VDB, `manifest.json`, PNG-превью и скрипт импорта в Blender. Вместо симуляции пока аналитический тестовый шар: он нужен только чтобы проверить весь путь от программы до рендера в Blender.

## Сборка на Windows

Нужно:
- Visual Studio 2022 с компонентом «Desktop development with C++»;
- CUDA Toolkit 13.x (ставить после Visual Studio, чтобы установилась интеграция с VS);
- CMake 3.25+ (входит в Visual Studio);
- Git;
- vcpkg.

### 1. vcpkg (один раз)

```bat
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
setx VCPKG_ROOT C:\vcpkg
```

После `setx` перезапусти терминал.

### 2. Конфигурация и сборка

Команды выполняются в корне репозитория (`cmd` или PowerShell):

```bat
cmake --preset windows
cmake --build --preset windows
ctest --preset windows
```

Первый `cmake --preset windows` сам скачает и соберёт зависимости через vcpkg (OpenVDB, TBB, Blosc, Boost и т.д.). Это **долго, 30–60 минут**, но только один раз.

Если CMake пишет, что не находит CUDA toolset для Visual Studio, переустанови CUDA Toolkit с галочкой «Visual Studio Integration». Временно можно собрать без GPU: `cmake --preset windows-cpu`.

### 3. Запуск

```bat
build\windows\Release\nukesim.exe --info
build\windows\Release\nukesim.exe scenes\test_fireball.toml
```

`--info` должен показать твою RTX 3060. Тестовая сцена пишет 120 кадров в `out\test_fireball\`: файлы `.vdb`, `manifest.json` и превью в `preview\`.

Опции:

| Опция | Что делает |
|---|---|
| `--device cpu\|cuda\|auto` | на чём считать |
| `--out DIR` | папка вывода |
| `--frames N` | только первые N кадров |
| `--preview-every N` | PNG-превью каждые N кадров (0 = выкл.) |

### 4. Blender (5.2)

```bat
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --python tools\blender\import_explosion.py -- --dir out\test_fireball --preview-scene
```

Скрипт:
- импортирует последовательность VDB как Volume-объект;
- собирает материал: плотность, излучение абсолютно чёрного тела по `temperature`, яркость по `flame`;
- включает motion blur по сетке `velocity`;
- добавляет точечный свет, который следует за шаром (позиция и яркость из `manifest.json`).

`--preview-scene` дополнительно ставит камеру, солнце, небо и плоскость земли. Без него импортируется только сам взрыв, в твою существующую сцену.

Остальные опции (`--density`, `--emission`, `--light-power`, …) описаны в начале скрипта.

## Что прислать после первой сборки

1. Вывод `nukesim.exe --info`.
2. Результат `ctest --preset windows`: строки с `test cases` и `assertions`.
3. Последние строки лога тестовой сцены, где видны время и размер файлов.
4. Скриншот из Blender. Если что-то упало, пришли текст ошибки целиком.

## Сборка на Linux

Нужны пакеты `libopenvdb-dev libtomlplusplus-dev doctest-dev libstb-dev libboost-iostreams-dev`.

```sh
cmake --preset linux-cpu && cmake --build --preset linux-cpu && ctest --preset linux-cpu
```

## Структура

```
src/core/      бэкенд CPU/CUDA (parallel_for), память, сетки, шум, расписание кадров
src/io/        конфиг сцены (TOML), экспорт OpenVDB, manifest.json
src/preview/   быстрые PNG-превью
src/scenes/    сцены (пока только test_fireball)
src/app/       запуск сцены, CLI
scenes/        конфиги сцен
tools/blender/ импорт в Blender
tests/         юнит-тесты (doctest)
```

Код, который работает и на CPU, и на GPU, лежит в файлах из списка `SIM_KERNEL_SOURCES` в `CMakeLists.txt`. В CUDA-сборке их компилирует nvcc, в CPU-сборке обычный компилятор. Поэтому каждое ядро пишется один раз.
