"""
Usage:
    from nmap_parser import NmapParser
    parser = NmapParser("output.xml")
    result = parser.parse()
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# Data Models
@dataclass
class Service:
    port: int
    protocol: str          # tcp / udp
    state: str             # open / closed / filtered
    name: str              # e.g. "http", "ssh", "smb"
    product: str           # e.g. "Apache httpd"
    version: str           # e.g. "2.4.51"
    extra_info: str        # any extra banner info
    tunnel: str            # e.g. "ssl"
    cpe: list[str] = field(default_factory=list)   # Common Platform Enumeration
    scripts: dict = field(default_factory=dict)     # NSE script id → output


@dataclass
class Host:
    ip: str
    hostname: str
    state: str             # up / down
    os_matches: list[dict] = field(default_factory=list)   # [{name, accuracy}]
    services: list[Service] = field(default_factory=list)
    traceroute: list[str] = field(default_factory=list)


@dataclass
class NmapResult:
    command: str           # nmap command that was run
    scan_start: str
    scan_end: str
    hosts: list[Host] = field(default_factory=list)

    # Convenience accessors
    def open_ports(self) -> list[dict]:
        """Flat list of all open ports across all hosts."""
        ports = []
        for host in self.hosts:
            for svc in host.services:
                if svc.state == "open":
                    ports.append({
                        "ip": host.ip,
                        "port": svc.port,
                        "protocol": svc.protocol,
                        "service": svc.name,
                        "product": svc.product,
                        "version": svc.version,
                    })
        return ports

    def services_by_name(self, name: str) -> list[dict]:
        """Find all instances of a service by name (e.g. 'http', 'ssh')."""
        return [p for p in self.open_ports() if p["service"] == name]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Compact summary string for LLM context injection (~200 tokens)."""
        lines = [f"Scan: {self.command}"]
        for host in self.hosts:
            lines.append(f"\nHost: {host.ip} ({host.hostname}) — {host.state}")
            if host.os_matches:
                top_os = host.os_matches[0]
                lines.append(f"  OS: {top_os['name']} ({top_os['accuracy']}% confidence)")
            open_svcs = [s for s in host.services if s.state == "open"]
            for svc in open_svcs:
                version_str = f"{svc.product} {svc.version}".strip()
                lines.append(
                    f"  {svc.port}/{svc.protocol}  {svc.name:<12}  {version_str}"
                )
                for script_id, output in svc.scripts.items():
                    # Truncate long script output for summary
                    short = output.strip().replace("\n", " ")[:120]
                    lines.append(f"    [NSE:{script_id}] {short}")
        return "\n".join(lines)


# Parser
class NmapParser:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path

    def parse(self) -> NmapResult:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        result = NmapResult(
            command=root.attrib.get("args", ""),
            scan_start=root.attrib.get("startstr", ""),
            scan_end=self._get_scan_end(root),
        )

        for host_elem in root.findall("host"):
            host = self._parse_host(host_elem)
            if host:
                result.hosts.append(host)

        return result

    #Private helpers

    def _get_scan_end(self, root: ET.Element) -> str:
        runstats = root.find("runstats/finished")
        if runstats is not None:
            return runstats.attrib.get("timestr", "")
        return ""

    def _parse_host(self, host_elem: ET.Element) -> Optional[Host]:
        # Host state
        status = host_elem.find("status")
        if status is None:
            return None
        state = status.attrib.get("state", "unknown")

        # IP address
        ip = ""
        hostname = ""
        for addr in host_elem.findall("address"):
            if addr.attrib.get("addrtype") in ("ipv4", "ipv6"):
                ip = addr.attrib.get("addr", "")

        # Hostnames
        hostnames_elem = host_elem.find("hostnames")
        if hostnames_elem is not None:
            hn = hostnames_elem.find("hostname")
            if hn is not None:
                hostname = hn.attrib.get("name", "")

        host = Host(ip=ip, hostname=hostname, state=state)

        # OS detection
        os_elem = host_elem.find("os")
        if os_elem is not None:
            for match in os_elem.findall("osmatch"):
                host.os_matches.append({
                    "name": match.attrib.get("name", ""),
                    "accuracy": int(match.attrib.get("accuracy", 0)),
                })
            # Sort by accuracy descending
            host.os_matches.sort(key=lambda x: x["accuracy"], reverse=True)

        # Ports / services
        ports_elem = host_elem.find("ports")
        if ports_elem is not None:
            for port_elem in ports_elem.findall("port"):
                svc = self._parse_port(port_elem)
                host.services.append(svc)

        # Traceroute (optional)
        trace_elem = host_elem.find("trace")
        if trace_elem is not None:
            for hop in trace_elem.findall("hop"):
                hop_ip = hop.attrib.get("ipaddr", "")
                if hop_ip:
                    host.traceroute.append(hop_ip)

        return host

    def _parse_port(self, port_elem: ET.Element) -> Service:
        port_id = int(port_elem.attrib.get("portid", 0))
        protocol = port_elem.attrib.get("protocol", "tcp")

        # State
        state_elem = port_elem.find("state")
        state = state_elem.attrib.get("state", "unknown") if state_elem is not None else "unknown"

        # Service info
        svc_elem = port_elem.find("service")
        name = product = version = extra_info = tunnel = ""
        cpe_list = []

        if svc_elem is not None:
            name = svc_elem.attrib.get("name", "")
            product = svc_elem.attrib.get("product", "")
            version = svc_elem.attrib.get("version", "")
            extra_info = svc_elem.attrib.get("extrainfo", "")
            tunnel = svc_elem.attrib.get("tunnel", "")
            cpe_list = [
                cpe.text for cpe in svc_elem.findall("cpe") if cpe.text
            ]

        # NSE script output
        scripts = {}
        for script_elem in port_elem.findall("script"):
            script_id = script_elem.attrib.get("id", "")
            script_out = script_elem.attrib.get("output", "")
            # Also capture structured table output if present
            tables = script_elem.findall("table")
            if tables:
                structured = self._parse_script_tables(tables)
                scripts[script_id] = {
                    "output": script_out,
                    "structured": structured,
                }
            else:
                scripts[script_id] = script_out

        return Service(
            port=port_id,
            protocol=protocol,
            state=state,
            name=name,
            product=product,
            version=version,
            extra_info=extra_info,
            tunnel=tunnel,
            cpe=cpe_list,
            scripts=scripts,
        )

    def _parse_script_tables(self, tables: list[ET.Element]) -> list[dict]:
        """Recursively parse NSE <table> elements into Python dicts."""
        result = []
        for table in tables:
            entry = {}
            for elem in table.findall("elem"):
                key = elem.attrib.get("key", "")
                value = elem.text or ""
                if key:
                    entry[key] = value
                else:
                    entry.setdefault("_values", []).append(value)
            # Nested tables
            nested = self._parse_script_tables(table.findall("table"))
            if nested:
                entry["_tables"] = nested
            if entry:
                result.append(entry)
        return result


# CLI smoke test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nmap_parser.py <scan.xml>")
        sys.exit(1)

    parser = NmapParser(sys.argv[1])
    result = parser.parse()

    print("=" * 60)
    print("SUMMARY (LLM context injection preview)")
    print("=" * 60)
    print(result.summary())

    print("\n" + "=" * 60)
    print("OPEN PORTS (flat list)")
    print("=" * 60)
    for p in result.open_ports():
        print(f"  {p['ip']}:{p['port']}/{p['protocol']}  {p['service']}  {p['product']} {p['version']}")

    print("\n" + "=" * 60)
    print("FULL JSON STATE")
    print("=" * 60)
    print(result.to_json())