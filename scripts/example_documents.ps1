# Requires: PowerShell 5+ (built-in on Windows 10/11)

$items = @(
    @{ Country="Belarus"; Lang="be"; File="pravily-darozhnaga-ruhu-respubliki-belarus.pdf"; Url="https://knihi-online.com/assets/files/24/10/pravily-darozhnaga-ruhu-respubliki-belarus.pdf" },
    @{ Country="Czechia"; Lang="cs"; File="zakon-c-361_2000-verze-BESIP-k-1-1-24.pdf"; Url="https://autoskolaolda.cz/wp-content/uploads/2024/01/zakon-c-361_2000-verze-BESIP-k-1-1-24.pdf" },
    @{ Country="UK";     Lang="en"; File="the_official_highway_code_-_10-04-2025_2.pdf"; Url="https://www.highwaycodeuk.co.uk/uploads/3/2/9/2/3292309/the_official_highway_code_-_10-04-2025_2.pdf" }
)

foreach ($i in $items) {
    $dir = Join-Path "documents" (Join-Path $i.Country $i.Lang)
    New-Item -ItemType Directory -Path $dir -Force | Out-Null

    $outfile = Join-Path $dir $i.File
    Invoke-WebRequest -Uri $i.Url -OutFile $outfile -UseBasicParsing
}
